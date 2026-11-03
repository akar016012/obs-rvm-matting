// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Aditya Kar
//
// Person Matting (RVM): an OBS video filter that cuts a person out of a camera
// feed with the Robust Video Matting model running through Core ML.
//
// OBS runs async video filters on its graphics loop, so filter_video only
// copies the frame into a CVPixelBuffer and hands it to a background queue.
// The queue runs the model, steadies the matte where the picture is still,
// and keeps the last few mattes tagged with their frame's timestamp.
// video_render multiplies the source's alpha by a matte on the GPU.
//
// With "line up cutout with video" on, filter_video holds each frame until
// its own matte is ready, so the video is delayed by about one frame and
// moving hands stay inside the cutout. With it off, frames pass straight
// through and the newest matte trails them by one inference.

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#import <VideoToolbox/VideoToolbox.h>
#import <Accelerate/Accelerate.h>

#include <obs-module.h>
#include <util/platform.h>
#include <util/threading.h>
#include <graphics/vec3.h>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-rvm-matting", "en-US")

#define PLUGIN_VERSION "1.0.0"

#define blog_rvm(level, fmt, ...) blog(level, "[rvm-matting] " fmt, ##__VA_ARGS__)

// Recurrent states are reset when frames stop arriving for this long, so a
// paused or reconnected camera does not start from stale memory.
static const double kStateResetGapSeconds = 1.0;
// Inference timings are summarized in the OBS log at this interval.
static const double kStatsIntervalSeconds = 30.0;
// A requested matte snapshot waits this many mattes, so the model's
// recurrent state has settled before the image is written.
static const int kSnapshotDelayMattes = 90;
// Edge smoothing compares consecutive camera frames on a grid this many
// times smaller than the model input. Where the picture is still, changes in
// the matte are the model wavering and are damped; where the picture moves,
// the matte follows at once.
static const uint32_t kMotionGridScale = 4;
// Luma differences up to this many levels (out of 255) are camera noise.
static const float kMotionNoiseFloor = 3.0f;
// A luma difference this far above the noise floor counts as full movement.
// Keep it small: if movement is only partly passed through, the parts of an
// edge that move catch up at different speeds and the outline turns wavy.
static const float kMotionThreshold = 12.0f;
// The model's edge carries the camera's sensor noise, which is strongest in
// dark areas such as headphone foam. The matte is blurred over this many
// pixels (odd, tent-shaped) to average it out before steadying.
static const vImagePixelCount kEdgeBlur = 5;
// Movement is spread this many grid cells (odd) around where it is seen,
// so an edge that moves into still background is not held back.
static const vImagePixelCount kMotionSpread = 7;
// Recent mattes are kept with the timestamp of the frame they came from, so
// the renderer can pair each frame with its own matte.
#define MATTE_SLOTS 4
// With "line up cutout with video", at most this many frames wait for their
// matte. If the model falls further behind, the oldest frame is shown anyway.
#define MAX_HELD_FRAMES 4
// When no matte has arrived for this long, frames pass straight through
// instead of waiting, so a failing model never freezes or delays the video.
static const uint64_t kSyncGiveUpNs = 500000000ULL;
// At most this many finished mattes may wait for the publish queue; while
// they do, new frames are not sent to the model.
static const int kMaxPostPending = 2;
// A photo of the empty background is compared with each frame. Where the
// average color difference is at most kPlateMatchLow levels the pixel is
// background and is cut out; from kPlateMatchHigh up it is left to the model.
static const float kPlateMatchLow = 10.0f;
static const float kPlateMatchHigh = 28.0f;
// The color difference is averaged over this many pixels (odd) to ignore
// camera noise.
static const vImagePixelCount kPlateBlur = 5;
// The solid middle of the person, shrunk by this many motion-grid cells
// (odd), is never cut by the photo or by chair removal, so clothes that
// match what is behind them keep no holes.
static const vImagePixelCount kCoreShrink = 9;
// Where the model is sure it sees background and nothing moves, the photo
// drifts toward the live picture by 1/16 per matte, following lighting.
static const int kPlateAdaptShift = 4;
// The photo is taken this long after the capture button is clicked.
static const double kPlateCaptureDelaySeconds = 5.0;
// Chair removal: a color counts as chair once (red - green) / red reaches
// kChairRednessLow and fully at kChairRednessHigh, and only when red is at
// least kChairMinRed, so the black rim and shadows are not chair on their own.
static const float kChairRednessLow = 0.35f;
static const float kChairRednessHigh = 0.5f;
static const float kChairMinRed = 25.0f;
// ... and once (blue - green) / red reaches kChairMagentaLow, fully at
// kChairMagentaHigh.
static const float kChairMagentaLow = 0.0f;
static const float kChairMagentaHigh = 0.1f;
// The frame is blurred over this many pixels (odd) before scoring, so the
// mesh's black threads blend into its red.
static const vImagePixelCount kChairBlur = 11;
// The cut is widened by this many pixels (odd) to take the chair's rim.
static const vImagePixelCount kChairRim = 9;
// Pixels the model counts as mostly the person (at least this matte value)
// are judged on a frame blurred over only kChairFineBlur pixels (odd), so
// they are cut only if they look like chair themselves. This keeps the soft
// edge of a black earcup in front of the chair from being nibbled away.
static const uint8_t kChairSureMatte = 128;
static const vImagePixelCount kChairFineBlur = 3;
// Within this many pixels (odd) of the cut, strips of the matte thinner
// than kChairSliver pixels (odd) are removed, which takes the leftover rim.
static const vImagePixelCount kChairCleanupReach = 41;
static const vImagePixelCount kChairSliver = 9;
// Automatic color: every kAutoColorEvery mattes the frame is measured to set
// new targets, and on every matte the color moves kAutoColorRate of the way
// toward them. Small steps on every frame settle over a few seconds without
// any visible jumps.
static const int kAutoColorEvery = 15;
static const float kAutoColorRate = 0.02f;
// Exposure is set from the background's highlights, never from the person,
// so every skin tone keeps its own brightness: the background's
// 98th-percentile brightness (in linear light) is brought to
// kAutoHighlightTarget, within kAutoGainMin..Max.
static const float kAutoHighlightTarget = 0.80f;
static const float kAutoGainMin = 0.7f;
static const float kAutoGainMax = 3.0f;
// White balance makes plain background areas (walls) neutral. It needs at
// least this share of the measured pixels, and each channel's correction
// stays within kAutoBalanceMin..Max.
static const float kAutoBalanceMinShare = 0.005f;
static const float kAutoBalanceMin = 0.75f;
static const float kAutoBalanceMax = 1.33f;
// After the gain, mid-tones are lifted with this power in linear light, much
// like a phone camera's tone curve.
static const float kAutoColorLift = 0.8f;
// Exposure is measured on the background when it makes up at least this
// share of the frame, and on the whole frame otherwise.
static const float kAutoBackgroundMinShare = 0.05f;

// Both models take 1920x1080 frames and return a 1920x1080 matte; see
// tools/convert_model.py.
static NSString *const kModelFast = @"rvm_alpha_mobilenetv3_1920x1080_s0.25.mlmodelc";
static NSString *const kModelAccurate = @"rvm_alpha_resnet50_1920x1080_s0.25.mlmodelc";

// Flicker along the edge where the picture is still: the change per edge
// pixel between consecutive mattes as the model returned them (raw) and as
// shown, summed over the stats interval. This is the "dancing edge" measure.
struct edge_flicker {
	double raw;
	double shown;
	uint64_t pixels;
};

#pragma mark - Matting engine

@interface RVMEngine : NSObject
@property (atomic) BOOL busy;
@property (atomic) BOOL canceled;
// 0 turns the steadying off; 1 damps edge flicker the most.
@property (atomic) float stability;
// Timestamp of the frame behind the newest matte, or 0 before the first.
@property (atomic) uint64_t latestMatteTimestamp;
// The load request that created this engine, for the filter's bookkeeping.
@property (atomic) uint64_t request;
// Whether the empty-background photo, if there is one, is used.
@property (atomic) BOOL usePlate;
// Whether a red chair behind the person is always cut out.
@property (atomic) BOOL removeChair;
// Whether exposure and white balance are corrected automatically.
@property (atomic) BOOL autoColor;
// The current automatic color state: red, green and blue gains in linear
// light, then the mid-tone lift. It fades to 1, 1, 1, 1 while automatic
// color is off.
- (void)getColorState:(float *)state;
// Starts from another engine's state, so a model change keeps the color.
// Call it before the engine is given any frames.
- (void)setColorState:(const float *)state;
// The photo is loaded from and saved to platePath.
- (instancetype)initWithModelPath:(NSString *)path
		     computeUnits:(MLComputeUnits)units
			platePath:(NSString *)platePath;
// Takes the empty-background photo from the first frame after this many
// seconds, so there is time to step out of view.
- (void)requestPlateCaptureIn:(double)seconds;
- (void)requestPlateForget;
// Whether frames in this frame's pixel format can be sent to the model.
- (BOOL)acceptsFrame:(const struct obs_source_frame *)frame;
- (BOOL)submitFrame:(const struct obs_source_frame *)frame;
// Copies the newest matte made from a frame no later than timestamp into dst,
// unless that matte is the one already copied (*generation). Each pixel is
// two bytes: the matte, then the chair spill strength.
- (BOOL)copyMatteFor:(uint64_t)timestamp
	  generation:(uint64_t *)generation
		into:(uint8_t **)dst
	       width:(uint32_t *)width
	      height:(uint32_t *)height;
- (void)requestSnapshotTo:(NSString *)path;
- (void)drain;
@end

@implementation RVMEngine {
	// _queue runs the model; _postQueue measures motion, smooths and
	// publishes each matte, so the model can start on the next frame
	// straight away. Each ivar below belongs to one of the two queues.
	dispatch_queue_t _queue;
	dispatch_queue_t _postQueue;
	BOOL _postResetPending;
	// Mattes handed to the publish queue and not yet finished.
	_Atomic int _postPending;
	BOOL _loggedFrameFormat;
	MLModel *_model;
	NSUInteger _modelWidth;
	NSUInteger _modelHeight;
	NSDictionary<NSString *, MLMultiArray *> *_zeroStates;
	NSMutableDictionary<NSString *, MLFeatureValue *> *_states;
	VTPixelTransferSessionRef _transfer;
	CVPixelBufferPoolRef _inputPool;
	OSType _inputPoolFormat;
	uint32_t _inputPoolWidth;
	uint32_t _inputPoolHeight;
	CVPixelBufferPoolRef _bgraPool;
	double _lastFrameTime;

	NSLock *_matteLock;
	uint8_t *_slots[MATTE_SLOTS];
	uint64_t _slotTimestamps[MATTE_SLOTS];
	uint64_t _slotGenerations[MATTE_SLOTS]; // 0 marks an empty slot
	uint32_t _matteWidth;
	uint32_t _matteHeight;
	uint64_t _matteGeneration;

	// Owned by the publish queue: the model's raw matte and the smoothed
	// matte, kept in float so slow changes are not lost to rounding.
	uint8_t *_incoming;
	uint8_t *_previous;
	uint8_t *_edgeBlurred;
	float *_smoothed;
	uint32_t _smoothWidth;
	uint32_t _smoothHeight;
	BOOL _smoothValid;

	// Owned by the publish queue: the camera frame's luma on the motion
	// grid, the previous frame's luma, and the spread difference between them.
	uint8_t *_gridBGRA;
	uint8_t *_luma;
	uint8_t *_prevLuma;
	uint8_t *_diff;
	uint8_t *_motion;
	uint32_t _gridWidth;
	uint32_t _gridHeight;
	BOOL _motionValid;

	// Owned by the publish queue: the empty-background photo (BGRA at the
	// model size) and the blurred color difference to it.
	NSString *_platePath;
	uint8_t *_plate;
	uint8_t *_plateDiff;
	uint8_t *_plateDiffBlur;
	BOOL _plateValid;

	// Owned by the publish queue: the person's solid middle on the motion
	// grid, rebuilt from each raw matte by updateCoreFrom.
	uint8_t *_coreSource;
	uint8_t *_core;
	uint32_t _coreWidth;
	uint32_t _coreHeight;

	// Set from the OBS video thread, read on the publish queue.
	_Atomic double _plateCaptureAt;
	_Atomic bool _plateForgetRequested;

	// Owned by the publish queue: chair removal buffers. _chairCut holds how
	// much of each pixel was cut from the latest matte while _chairMaskValid
	// is set.
	uint8_t *_chairBlur;
	uint8_t *_chairFine;
	uint8_t *_chairScore;
	uint8_t *_chairCut;
	uint8_t *_chairNear;
	uint8_t *_chairOpened;
	BOOL _chairMaskValid;

	NSString *_snapshotPath;
	int _snapshotCountdown;
	BOOL _loggedFormatError;
	double _statsStart;
	double _statsTotalMs;
	double _statsMaxMs;
	uint64_t _statsCount;
	// Incremented on the OBS video thread, read and reset on the publish
	// queue.
	_Atomic uint64_t _statsSkipped;
	// Mean absolute change per pixel between consecutive mattes, before and
	// after smoothing, summed over the stats interval.
	double _statsRawChange;
	double _statsChange;
	struct edge_flicker _statsEdge;

	// Automatic color. The gains and the lift are smoothed on the publish
	// queue and read by the renderer under _colorLock.
	int _autoColorCountdown;
	float _colorTarget[4]; // publish queue only: gains, then lift
	float _colorGains[3];
	float _colorLift;
	NSLock *_colorLock;
}

- (instancetype)initWithModelPath:(NSString *)path
		     computeUnits:(MLComputeUnits)units
			platePath:(NSString *)platePath
{
	if (!(self = [super init]))
		return nil;

	_queue = dispatch_queue_create("rvm-matting.inference", DISPATCH_QUEUE_SERIAL);
	_postQueue = dispatch_queue_create("rvm-matting.publish", DISPATCH_QUEUE_SERIAL);
	_matteLock = [NSLock new];
	_colorLock = [NSLock new];
	_colorGains[0] = _colorGains[1] = _colorGains[2] = 1.0f;
	_colorLift = 1.0f;
	_colorTarget[0] = _colorTarget[1] = _colorTarget[2] = _colorTarget[3] = 1.0f;

	MLModelConfiguration *config = [MLModelConfiguration new];
	config.computeUnits = units;
	NSError *error = nil;
	_model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path] configuration:config error:&error];
	if (!_model) {
		blog_rvm(LOG_ERROR, "Could not load model %s: %s", path.UTF8String,
			 error.localizedDescription.UTF8String);
		return nil;
	}

	MLImageConstraint *src = _model.modelDescription.inputDescriptionsByName[@"src"].imageConstraint;
	_modelWidth = src.pixelsWide;
	_modelHeight = src.pixelsHigh;

	NSMutableDictionary *zeros = [NSMutableDictionary dictionary];
	for (NSString *name in @[@"r1i", @"r2i", @"r3i", @"r4i"]) {
		MLMultiArrayConstraint *c = _model.modelDescription.inputDescriptionsByName[name].multiArrayConstraint;
		MLMultiArray *array = [[MLMultiArray alloc] initWithShape:c.shape dataType:c.dataType error:&error];
		if (!array) {
			blog_rvm(LOG_ERROR, "Could not allocate state %s", name.UTF8String);
			return nil;
		}
		[array getMutableBytesWithHandler:^(void *bytes, NSInteger size, NSArray<NSNumber *> *strides) {
			(void)strides;
			memset(bytes, 0, (size_t)size);
		}];
		zeros[name] = array;
	}
	_zeroStates = zeros;
	[self resetStates];

	if (VTPixelTransferSessionCreate(kCFAllocatorDefault, &_transfer) != noErr) {
		blog_rvm(LOG_ERROR, "Could not create pixel transfer session");
		return nil;
	}

	NSDictionary *bgraAttrs = @{
		(id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
		(id)kCVPixelBufferWidthKey: @(_modelWidth),
		(id)kCVPixelBufferHeightKey: @(_modelHeight),
		(id)kCVPixelBufferIOSurfacePropertiesKey: @{},
	};
	CVPixelBufferPoolCreate(kCFAllocatorDefault, NULL, (__bridge CFDictionaryRef)bgraAttrs, &_bgraPool);

	_platePath = [platePath copy];
	[self loadPlate];

	_statsStart = CFAbsoluteTimeGetCurrent();
	blog_rvm(LOG_INFO, "Loaded %s (%lux%lu, compute units %ld)", path.lastPathComponent.UTF8String,
		 (unsigned long)_modelWidth, (unsigned long)_modelHeight, (long)units);
	return self;
}

- (void)dealloc
{
	if (_transfer) {
		VTPixelTransferSessionInvalidate(_transfer);
		CFRelease(_transfer);
	}
	if (_inputPool)
		CVPixelBufferPoolRelease(_inputPool);
	if (_bgraPool)
		CVPixelBufferPoolRelease(_bgraPool);
	for (int s = 0; s < MATTE_SLOTS; s++)
		free(_slots[s]);
	free(_incoming);
	free(_previous);
	free(_edgeBlurred);
	free(_smoothed);
	free(_gridBGRA);
	free(_luma);
	free(_prevLuma);
	free(_diff);
	free(_motion);
	free(_plate);
	free(_plateDiff);
	free(_plateDiffBlur);
	free(_coreSource);
	free(_core);
	free(_chairBlur);
	free(_chairFine);
	free(_chairScore);
	free(_chairCut);
	free(_chairNear);
	free(_chairOpened);
}

// Runs on the model queue. The publish queue restarts its smoothing and
// motion history when it sees _postResetPending on the next matte.
- (void)resetStates
{
	_postResetPending = YES;
	_states = [NSMutableDictionary dictionary];
	for (NSString *name in _zeroStates)
		_states[name] = [MLFeatureValue featureValueWithMultiArray:_zeroStates[name]];
}

static OSType pixel_format_for(enum video_format format, bool full_range)
{
	switch (format) {
	case VIDEO_FORMAT_NV12:
		return full_range ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
				  : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
	case VIDEO_FORMAT_YUY2:
		return kCVPixelFormatType_422YpCbCr8_yuvs;
	case VIDEO_FORMAT_UYVY:
		return kCVPixelFormatType_422YpCbCr8;
	case VIDEO_FORMAT_BGRA:
	case VIDEO_FORMAT_BGRX:
		return kCVPixelFormatType_32BGRA;
	default:
		return 0;
	}
}

static void copy_plane(uint8_t *dst, size_t dst_stride, const uint8_t *src, size_t src_stride, size_t row_bytes,
		       size_t rows)
{
	if (dst_stride == src_stride && src_stride == row_bytes) {
		memcpy(dst, src, row_bytes * rows);
		return;
	}
	for (size_t y = 0; y < rows; y++)
		memcpy(dst + y * dst_stride, src + y * src_stride, row_bytes);
}

// Copies an OBS frame into a pooled CVPixelBuffer. Runs on the OBS graphics
// loop, so it does nothing slower than a memcpy.
- (CVPixelBufferRef)copyPixelBufferFromFrame:(const struct obs_source_frame *)frame CF_RETURNS_RETAINED
{
	OSType format = pixel_format_for(frame->format, frame->full_range);
	if (!format)
		return NULL;

	if (!_inputPool || format != _inputPoolFormat || frame->width != _inputPoolWidth ||
	    frame->height != _inputPoolHeight) {
		if (_inputPool)
			CVPixelBufferPoolRelease(_inputPool);
		_inputPool = NULL;
		NSDictionary *attrs = @{
			(id)kCVPixelBufferPixelFormatTypeKey: @(format),
			(id)kCVPixelBufferWidthKey: @(frame->width),
			(id)kCVPixelBufferHeightKey: @(frame->height),
			(id)kCVPixelBufferIOSurfacePropertiesKey: @{},
		};
		if (CVPixelBufferPoolCreate(kCFAllocatorDefault, NULL, (__bridge CFDictionaryRef)attrs, &_inputPool) !=
		    kCVReturnSuccess)
			return NULL;
		_inputPoolFormat = format;
		_inputPoolWidth = frame->width;
		_inputPoolHeight = frame->height;
	}

	CVPixelBufferRef buffer = NULL;
	if (CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, _inputPool, &buffer) != kCVReturnSuccess)
		return NULL;

	CVPixelBufferLockBaseAddress(buffer, 0);
	if (format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ||
	    format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange) {
		copy_plane(CVPixelBufferGetBaseAddressOfPlane(buffer, 0), CVPixelBufferGetBytesPerRowOfPlane(buffer, 0),
			   frame->data[0], frame->linesize[0], frame->width, frame->height);
		copy_plane(CVPixelBufferGetBaseAddressOfPlane(buffer, 1), CVPixelBufferGetBytesPerRowOfPlane(buffer, 1),
			   frame->data[1], frame->linesize[1], (frame->width + 1) / 2 * 2, (frame->height + 1) / 2);
	} else {
		size_t bytes_per_pixel = format == kCVPixelFormatType_32BGRA ? 4 : 2;
		copy_plane(CVPixelBufferGetBaseAddress(buffer), CVPixelBufferGetBytesPerRow(buffer), frame->data[0],
			   frame->linesize[0], frame->width * bytes_per_pixel, frame->height);
	}
	CVPixelBufferUnlockBaseAddress(buffer, 0);

	CVBufferSetAttachment(buffer, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2,
			      kCVAttachmentMode_ShouldPropagate);
	return buffer;
}

- (BOOL)acceptsFrame:(const struct obs_source_frame *)frame
{
	return pixel_format_for(frame->format, frame->full_range) != 0;
}

- (BOOL)submitFrame:(const struct obs_source_frame *)frame
{
	if (self.canceled)
		return NO;
	if (![self acceptsFrame:frame]) {
		if (!_loggedFrameFormat)
			blog_rvm(LOG_WARNING, "Camera frames in OBS video format %d are not supported; set the "
					      "camera to NV12, YUY2, UYVY or BGRA",
				 (int)frame->format);
		_loggedFrameFormat = YES;
		return NO;
	}
	// Drop frames while the model or the publish queue is busy, so the matte
	// always follows the newest frame instead of working through a backlog.
	if (self.busy || _postPending >= kMaxPostPending) {
		_statsSkipped++;
		return NO;
	}

	CVPixelBufferRef input = [self copyPixelBufferFromFrame:frame];
	if (!input)
		return NO;

	uint64_t timestamp = frame->timestamp;
	self.busy = YES;
	dispatch_async(_queue, ^{
		@autoreleasepool {
			[self runModelOn:input timestamp:timestamp];
		}
		CVPixelBufferRelease(input);
		self.busy = NO;
	});
	return YES;
}

// Measures where the picture moved since the previous frame, on a coarse
// grid, and spreads it a little so moving edges count as moving.
- (void)updateMotionFrom:(CVPixelBufferRef)bgra
{
	uint32_t gw = (uint32_t)(_modelWidth / kMotionGridScale);
	uint32_t gh = (uint32_t)(_modelHeight / kMotionGridScale);
	size_t count = (size_t)gw * gh;
	if (!_luma || gw != _gridWidth || gh != _gridHeight) {
		free(_gridBGRA);
		free(_luma);
		free(_prevLuma);
		free(_diff);
		free(_motion);
		_gridBGRA = malloc(count * 4);
		_luma = malloc(count);
		_prevLuma = malloc(count);
		_diff = malloc(count);
		_motion = malloc(count);
		_gridWidth = gw;
		_gridHeight = gh;
		_motionValid = NO;
	}

	CVPixelBufferLockBaseAddress(bgra, kCVPixelBufferLock_ReadOnly);
	vImage_Buffer src = {
		.data = CVPixelBufferGetBaseAddress(bgra),
		.height = _modelHeight,
		.width = _modelWidth,
		.rowBytes = CVPixelBufferGetBytesPerRow(bgra),
	};
	vImage_Buffer grid = {.data = _gridBGRA, .height = gh, .width = gw, .rowBytes = gw * 4};
	vImageScale_ARGB8888(&src, &grid, NULL, kvImageNoFlags);
	CVPixelBufferUnlockBaseAddress(bgra, kCVPixelBufferLock_ReadOnly);

	for (size_t i = 0; i < count; i++) {
		const uint8_t *p = _gridBGRA + i * 4;
		_luma[i] = (uint8_t)((29 * p[0] + 150 * p[1] + 77 * p[2]) >> 8);
	}

	if (_motionValid) {
		for (size_t i = 0; i < count; i++)
			_diff[i] = (uint8_t)abs((int)_luma[i] - (int)_prevLuma[i]);
		vImage_Buffer diff = {.data = _diff, .height = gh, .width = gw, .rowBytes = gw};
		vImage_Buffer motion = {.data = _motion, .height = gh, .width = gw, .rowBytes = gw};
		vImageMax_Planar8(&diff, &motion, NULL, 0, 0, kMotionSpread, kMotionSpread, kvImageEdgeExtend);
	} else {
		memset(_motion, 255, count);
		_motionValid = YES;
	}

	uint8_t *swap = _prevLuma;
	_prevLuma = _luma;
	_luma = swap;
}

#pragma mark Empty-background photo

- (void)requestPlateCaptureIn:(double)seconds
{
	_plateCaptureAt = CFAbsoluteTimeGetCurrent() + seconds;
}

- (void)requestPlateForget
{
	_plateForgetRequested = true;
}

- (void)allocatePlateBuffers
{
	size_t count = _modelWidth * _modelHeight;
	if (_plate)
		return;
	_plate = malloc(count * 4);
	_plateDiff = malloc(count);
	_plateDiffBlur = malloc(count);
}

// Marks the person's solid middle on the motion grid: cells that are fully
// opaque in the model's matte, shrunk inwards by kCoreShrink cells. Neither
// the photo nor chair removal ever cuts inside it.
- (void)updateCoreFrom:(const uint8_t *)matte width:(uint32_t)width height:(uint32_t)height
{
	uint32_t gw = width / kMotionGridScale;
	uint32_t gh = height / kMotionGridScale;
	if (!_core || gw != _coreWidth || gh != _coreHeight) {
		free(_coreSource);
		free(_core);
		_coreSource = malloc((size_t)gw * gh);
		_core = malloc((size_t)gw * gh);
		_coreWidth = gw;
		_coreHeight = gh;
	}
	for (uint32_t gy = 0; gy < gh; gy++) {
		const uint8_t *row = matte + (size_t)(gy * kMotionGridScale + kMotionGridScale / 2) * width;
		for (uint32_t gx = 0; gx < gw; gx++)
			_coreSource[gy * gw + gx] = row[gx * kMotionGridScale + kMotionGridScale / 2] >= 250 ? 255 : 0;
	}
	vImage_Buffer core_source = {.data = _coreSource, .height = gh, .width = gw, .rowBytes = gw};
	vImage_Buffer core = {.data = _core, .height = gh, .width = gw, .rowBytes = gw};
	vImageMin_Planar8(&core_source, &core, NULL, 0, 0, kCoreShrink, kCoreShrink, kvImageEdgeExtend);
}

// One adaptation step toward the live picture, rounded to the nearest level.
static inline int adapt_step(int d)
{
	const int half = 1 << (kPlateAdaptShift - 1);
	return (d + (d > 0 ? half : -half)) / (1 << kPlateAdaptShift);
}

static CGContextRef create_bgra_context(void *data, size_t width, size_t height)
{
	CGColorSpaceRef rgb = CGColorSpaceCreateDeviceRGB();
	CGContextRef ctx = CGBitmapContextCreate(data, width, height, 8, width * 4, rgb,
						 kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little);
	CGColorSpaceRelease(rgb);
	return ctx;
}

// Loads a saved photo, scaled to the model size. Runs before any frame.
- (void)loadPlate
{
	if (!_platePath.length || ![[NSFileManager defaultManager] fileExistsAtPath:_platePath])
		return;
	NSURL *url = [NSURL fileURLWithPath:_platePath];
	CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
	CGImageRef image = source ? CGImageSourceCreateImageAtIndex(source, 0, NULL) : NULL;
	if (source)
		CFRelease(source);
	if (!image) {
		blog_rvm(LOG_WARNING, "Could not read background photo %s", _platePath.UTF8String);
		return;
	}
	[self allocatePlateBuffers];
	CGContextRef ctx = create_bgra_context(_plate, _modelWidth, _modelHeight);
	if (ctx) {
		CGContextDrawImage(ctx, CGRectMake(0, 0, _modelWidth, _modelHeight), image);
		CGContextRelease(ctx);
		_plateValid = YES;
		blog_rvm(LOG_INFO, "Loaded background photo");
	}
	CGImageRelease(image);
}

- (void)savePlate
{
	if (!_platePath.length)
		return;
	CGContextRef ctx = create_bgra_context(_plate, _modelWidth, _modelHeight);
	CGImageRef image = ctx ? CGBitmapContextCreateImage(ctx) : NULL;
	if (image) {
		CGImageDestinationRef dest = CGImageDestinationCreateWithURL(
			(__bridge CFURLRef)[NSURL fileURLWithPath:_platePath],
			(__bridge CFStringRef)UTTypePNG.identifier,
			1, NULL);
		if (dest) {
			CGImageDestinationAddImage(dest, image, NULL);
			if (!CGImageDestinationFinalize(dest))
				blog_rvm(LOG_WARNING, "Could not save background photo to %s", _platePath.UTF8String);
			CFRelease(dest);
		}
		CGImageRelease(image);
	}
	if (ctx)
		CGContextRelease(ctx);
}

// Cuts out pixels that match the empty-background photo, except in the solid
// middle of the person, and lets the photo follow slow lighting changes.
// Runs on the publish queue with the model's raw matte, before smoothing.
- (void)applyPlateTo:(uint8_t *)matte width:(uint32_t)width height:(uint32_t)height frame:(CVPixelBufferRef)bgra
{
	if (_plateForgetRequested) {
		_plateForgetRequested = false;
		_plateValid = NO;
		if (_platePath.length)
			[[NSFileManager defaultManager] removeItemAtPath:_platePath error:nil];
		blog_rvm(LOG_INFO, "Forgot background photo");
	}
	if (width != _modelWidth || height != _modelHeight || CVPixelBufferGetWidth(bgra) != width ||
	    CVPixelBufferGetHeight(bgra) != height)
		return;

	CVPixelBufferLockBaseAddress(bgra, kCVPixelBufferLock_ReadOnly);
	const uint8_t *frame = CVPixelBufferGetBaseAddress(bgra);
	size_t stride = CVPixelBufferGetBytesPerRow(bgra);

	double capture_at = _plateCaptureAt;
	if (capture_at > 0 && CFAbsoluteTimeGetCurrent() >= capture_at) {
		_plateCaptureAt = 0;
		[self allocatePlateBuffers];
		for (uint32_t y = 0; y < height; y++)
			memcpy(_plate + (size_t)y * width * 4, frame + y * stride, (size_t)width * 4);
		_plateValid = YES;
		[self savePlate];
		blog_rvm(LOG_INFO, "Captured background photo");
	}
	if (!_plateValid || !self.usePlate) {
		CVPixelBufferUnlockBaseAddress(bgra, kCVPixelBufferLock_ReadOnly);
		return;
	}

	uint32_t gw = width / kMotionGridScale;
	uint32_t gh = height / kMotionGridScale;
	BOOL have_motion = _motionValid && _gridWidth == gw && _gridHeight == gh;

	// Color difference to the photo, and slow adaptation of the photo where
	// the model is sure this is background and the picture is still.
	for (uint32_t y = 0; y < height; y++) {
		const uint8_t *p = frame + y * stride;
		uint8_t *q = _plate + (size_t)y * width * 4;
		uint8_t *d = _plateDiff + (size_t)y * width;
		const uint8_t *m = matte + (size_t)y * width;
		const uint8_t *motion_row =
			have_motion ? _motion + (size_t)MIN(y / kMotionGridScale, gh - 1) * gw : NULL;
		for (uint32_t x = 0; x < width; x++, p += 4, q += 4) {
			int db = p[0] - q[0], dg = p[1] - q[1], dr = p[2] - q[2];
			d[x] = (uint8_t)((abs(db) + abs(dg) + abs(dr)) / 3);
			if (motion_row && m[x] < 8 &&
			    motion_row[MIN(x / kMotionGridScale, gw - 1)] <= (uint8_t)kMotionNoiseFloor) {
				q[0] = (uint8_t)(q[0] + adapt_step(db));
				q[1] = (uint8_t)(q[1] + adapt_step(dg));
				q[2] = (uint8_t)(q[2] + adapt_step(dr));
			}
		}
	}
	CVPixelBufferUnlockBaseAddress(bgra, kCVPixelBufferLock_ReadOnly);

	vImage_Buffer diff = {.data = _plateDiff, .height = height, .width = width, .rowBytes = width};
	vImage_Buffer blur = {.data = _plateDiffBlur, .height = height, .width = width, .rowBytes = width};
	vImageBoxConvolve_Planar8(&diff, &blur, NULL, 0, 0, kPlateBlur, kPlateBlur, 0, kvImageEdgeExtend);


	const float inv_range = 1.0f / (kPlateMatchHigh - kPlateMatchLow);
	for (uint32_t y = 0; y < height; y++) {
		uint8_t *m = matte + (size_t)y * width;
		const uint8_t *b = _plateDiffBlur + (size_t)y * width;
		const uint8_t *core_row = _core + (size_t)MIN(y / kMotionGridScale, gh - 1) * gw;
		for (uint32_t x = 0; x < width; x++) {
			if (!m[x] || b[x] >= kPlateMatchHigh || core_row[MIN(x / kMotionGridScale, gw - 1)])
				continue;
			float t = fmaxf(0.0f, ((float)b[x] - kPlateMatchLow) * inv_range);
			float keep = t * t * (3.0f - 2.0f * t);
			m[x] = (uint8_t)((float)m[x] * keep + 0.5f);
		}
	}
}

#pragma mark Chair keeping

// How strongly a color looks like the red chair mesh, from 0 to 1. Skin
// can be as red as the mesh under a webcam, but its red leans toward
// orange (more green than blue), while the mesh leans toward magenta (more
// blue than green). Wood leans orange too.
static inline float chair_score(int r, int g, int b)
{
	if (r <= 0)
		return 0.0f;
	float redness = ((float)(r - g) / (float)r - kChairRednessLow) / (kChairRednessHigh - kChairRednessLow);
	float magenta = ((float)(b - g) / (float)r - kChairMagentaLow) / (kChairMagentaHigh - kChairMagentaLow);
	float brightness = ((float)r - kChairMinRed) / 30.0f;
	return fminf(1.0f, fmaxf(0.0f, fminf(fminf(redness, magenta), brightness)));
}

// Removes the red chair: chair-colored pixels outside the person's solid
// middle are cut out, however sure the model is that they belong to the
// person. Runs on the publish queue with the raw matte, before smoothing.
- (void)applyChairTo:(uint8_t *)matte width:(uint32_t)width height:(uint32_t)height frame:(CVPixelBufferRef)bgra
{
	_chairMaskValid = NO;
	if (!self.removeChair || width != _modelWidth || height != _modelHeight ||
	    CVPixelBufferGetWidth(bgra) != width || CVPixelBufferGetHeight(bgra) != height)
		return;

	size_t count = (size_t)width * height;
	if (!_chairBlur) {
		_chairBlur = malloc(count * 4);
		_chairFine = malloc(count * 4);
		_chairScore = malloc(count);
		_chairCut = malloc(count);
		_chairNear = malloc(count);
		_chairOpened = malloc(count);
	}

	// Score a blurred copy of the frame, so the mesh's black threads blend
	// into its red and the mesh is removed as a whole.
	CVPixelBufferLockBaseAddress(bgra, kCVPixelBufferLock_ReadOnly);
	vImage_Buffer src = {
		.data = CVPixelBufferGetBaseAddress(bgra),
		.height = height,
		.width = width,
		.rowBytes = CVPixelBufferGetBytesPerRow(bgra),
	};
	vImage_Buffer blur = {.data = _chairBlur, .height = height, .width = width, .rowBytes = (size_t)width * 4};
	vImageBoxConvolve_ARGB8888(&src, &blur, NULL, 0, 0, kChairBlur, kChairBlur, NULL, kvImageEdgeExtend);
	// A lightly blurred copy judges single pixels where the model is sure.
	vImage_Buffer fine = {.data = _chairFine, .height = height, .width = width, .rowBytes = (size_t)width * 4};
	vImageBoxConvolve_ARGB8888(&src, &fine, NULL, 0, 0, kChairFineBlur, kChairFineBlur, NULL, kvImageEdgeExtend);
	CVPixelBufferUnlockBaseAddress(bgra, kCVPixelBufferLock_ReadOnly);

	const uint8_t *p = _chairBlur;
	for (size_t i = 0; i < count; i++, p += 4)
		_chairScore[i] = (uint8_t)(chair_score(p[2], p[1], p[0]) * 255.0f + 0.5f);
	// Widen the cut a little, so the chair's edge and black rim go with it.
	vImage_Buffer score = {.data = _chairScore, .height = height, .width = width, .rowBytes = width};
	vImage_Buffer cut = {.data = _chairCut, .height = height, .width = width, .rowBytes = width};
	vImageMax_Planar8(&score, &cut, NULL, 0, 0, kChairRim, kChairRim, kvImageEdgeExtend);

	uint32_t gw = _coreWidth, gh = _coreHeight;
	for (uint32_t y = 0; y < height; y++) {
		uint8_t *m = matte + (size_t)y * width;
		uint8_t *c = _chairCut + (size_t)y * width;
		const uint8_t *f = _chairFine + (size_t)y * width * 4;
		const uint8_t *core_row = _core + (size_t)MIN(y / kMotionGridScale, gh - 1) * gw;
		for (uint32_t x = 0; x < width; x++) {
			if (core_row[MIN(x / kMotionGridScale, gw - 1)]) {
				c[x] = 0;
				continue;
			}
			// A partial score already removes most of the pixel, so mesh
			// that scores only part way still disappears.
			int amount = MIN(255, (int)c[x] * 3 / 2);
			// Where the model leans toward the person, the wide blur is not
			// enough: red smeared onto a black earcup next to the chair
			// would bite into it. Such pixels are cut only as far as
			// they look like chair themselves. The mesh's black threads that
			// survive this are thin and go in the clean-up below.
			if (m[x] >= kChairSureMatte) {
				const uint8_t *q = f + (size_t)x * 4;
				int own = (int)(chair_score(q[2], q[1], q[0]) * 255.0f * 1.5f);
				amount = MIN(amount, MIN(255, own));
			}
			c[x] = (uint8_t)amount;
			m[x] = (uint8_t)((int)m[x] * (255 - amount) / 255);
		}
	}

	// The chair's black rim is not red, so thin strips of it survive next to
	// the removed mesh. Close to the cut, take away anything thinner than
	// kChairSliver pixels (a morphological opening); solid shapes such as a
	// headphone earcup are left as they are.
	vImage_Buffer near = {.data = _chairNear, .height = height, .width = width, .rowBytes = width};
	vImageMax_Planar8(&cut, &near, NULL, 0, 0, kChairCleanupReach, kChairCleanupReach, kvImageEdgeExtend);
	vImage_Buffer matte_buf = {.data = matte, .height = height, .width = width, .rowBytes = width};
	vImage_Buffer eroded = {.data = _chairScore, .height = height, .width = width, .rowBytes = width};
	vImage_Buffer opened = {.data = _chairOpened, .height = height, .width = width, .rowBytes = width};
	vImageMin_Planar8(&matte_buf, &eroded, NULL, 0, 0, kChairSliver, kChairSliver, kvImageEdgeExtend);
	vImageMax_Planar8(&eroded, &opened, NULL, 0, 0, kChairSliver, kChairSliver, kvImageEdgeExtend);
	for (uint32_t y = 0; y < height; y++) {
		uint8_t *m = matte + (size_t)y * width;
		const uint8_t *n = _chairNear + (size_t)y * width;
		const uint8_t *o = _chairOpened + (size_t)y * width;
		const uint8_t *core_row = _core + (size_t)MIN(y / kMotionGridScale, gh - 1) * gw;
		for (uint32_t x = 0; x < width; x++) {
			if (n[x] && o[x] < m[x] && !core_row[MIN(x / kMotionGridScale, gw - 1)])
				m[x] = o[x];
		}
	}
	_chairMaskValid = YES;
}

- (void)runModelOn:(CVPixelBufferRef)input timestamp:(uint64_t)timestamp
{
	if (self.canceled)
		return;

	double now = CFAbsoluteTimeGetCurrent();
	if (_lastFrameTime > 0 && now - _lastFrameTime > kStateResetGapSeconds)
		[self resetStates];
	_lastFrameTime = now;

	CVPixelBufferRef bgra = NULL;
	if (CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, _bgraPool, &bgra) != kCVReturnSuccess)
		return;
	if (VTPixelTransferSessionTransferImage(_transfer, input, bgra) != noErr) {
		CVPixelBufferRelease(bgra);
		return;
	}

	NSMutableDictionary *features = [_states mutableCopy];
	features[@"src"] = [MLFeatureValue featureValueWithPixelBuffer:bgra];
	NSError *error = nil;
	MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:features
												   error:&error];
	id<MLFeatureProvider> output = provider ? [_model predictionFromFeatures:provider error:&error] : nil;
	if (!output) {
		CVPixelBufferRelease(bgra);
		blog_rvm(LOG_WARNING, "Inference failed: %s", error.localizedDescription.UTF8String);
		[self resetStates];
		return;
	}

	for (NSString *name in @[@"r1", @"r2", @"r3", @"r4"]) {
		MLFeatureValue *value = [output featureValueForName:[name stringByAppendingString:@"o"]];
		if (value)
			_states[[name stringByAppendingString:@"i"]] = value;
	}

	CVPixelBufferRef pha = [output featureValueForName:@"pha"].imageBufferValue;
	double ms = (CFAbsoluteTimeGetCurrent() - now) * 1000.0;
	if (!pha) {
		CVPixelBufferRelease(bgra);
		return;
	}

	CVPixelBufferRetain(pha);
	BOOL reset = _postResetPending;
	_postResetPending = NO;
	_postPending++;
	dispatch_async(_postQueue, ^{
		if (reset) {
			self->_smoothValid = NO;
			self->_motionValid = NO;
		}
		if (!self.canceled) {
			[self updateMotionFrom:bgra];
			[self storeMatte:pha frame:bgra timestamp:timestamp];
			[self recordTiming:ms];
		}
		CVPixelBufferRelease(bgra);
		CVPixelBufferRelease(pha);
		self->_postPending--;
	});
}

// Damps changes in the matte where the camera picture is still. Each pixel
// moves toward the new value by a weight that is small where nothing moved
// (so the model's wavering settles) and 1 where the picture moved (so the
// cutout follows at once). raw and previous are the unblurred mattes of this
// frame and the last, used only for the edge flicker statistic. Returns the
// summed absolute change, for the overall flicker statistic.
static double smooth_matte(float *smoothed, const uint8_t *incoming, const uint8_t *raw, const uint8_t *previous,
			   uint32_t width, uint32_t height, const uint8_t *motion, uint32_t grid_width,
			   uint32_t grid_height, float stability, struct edge_flicker *edge)
{
	const float still_weight = 1.0f - 0.95f * stability;
	const float inv_threshold = 1.0f / kMotionThreshold;
	double change = 0.0;
	for (uint32_t y = 0; y < height; y++) {
		const uint8_t *motion_row = motion + (size_t)MIN(y / kMotionGridScale, grid_height - 1) * grid_width;
		float *s = smoothed + (size_t)y * width;
		const uint8_t *in = incoming + (size_t)y * width;
		const uint8_t *r = raw + (size_t)y * width;
		const uint8_t *p = previous + (size_t)y * width;
		for (uint32_t x = 0; x < width; x++) {
			float m = (float)motion_row[MIN(x / kMotionGridScale, grid_width - 1)];
			float moving = fminf(1.0f, fmaxf(0.0f, (m - kMotionNoiseFloor) * inv_threshold));
			float w = still_weight + (1.0f - still_weight) * moving;
			float d = (float)in[x] - s[x];
			s[x] += d * w;
			change += fabsf(d * w);
			if (moving == 0.0f && ((r[x] >= 8 && r[x] <= 247) || (p[x] >= 8 && p[x] <= 247))) {
				edge->raw += abs((int)r[x] - (int)p[x]);
				edge->shown += fabsf(d * w);
				edge->pixels++;
			}
		}
	}
	return change;
}

// Converts the model's matte to 8 bits per pixel, smooths it over time and
// publishes it for rendering, tagged with the timestamp of its frame. The
// converted models return 8-bit grayscale; half-float output is converted.
- (void)storeMatte:(CVPixelBufferRef)pha frame:(CVPixelBufferRef)bgra timestamp:(uint64_t)timestamp
{
	uint32_t width = (uint32_t)CVPixelBufferGetWidth(pha);
	uint32_t height = (uint32_t)CVPixelBufferGetHeight(pha);
	size_t count = (size_t)width * height;
	OSType format = CVPixelBufferGetPixelFormatType(pha);
	if (format != kCVPixelFormatType_OneComponent8 && format != kCVPixelFormatType_OneComponent16Half) {
		if (!_loggedFormatError)
			blog_rvm(LOG_ERROR, "Unsupported matte pixel format 0x%08x", (unsigned)format);
		_loggedFormatError = YES;
		return;
	}

	if (!_incoming || _smoothWidth != width || _smoothHeight != height) {
		free(_incoming);
		free(_previous);
		free(_edgeBlurred);
		free(_smoothed);
		_incoming = malloc(count);
		_previous = malloc(count);
		_edgeBlurred = malloc(count);
		_smoothed = malloc(count * sizeof(float));
		_smoothWidth = width;
		_smoothHeight = height;
		_smoothValid = NO;
	}

	CVPixelBufferLockBaseAddress(pha, kCVPixelBufferLock_ReadOnly);
	vImage_Buffer src = {
		.data = CVPixelBufferGetBaseAddress(pha),
		.height = height,
		.width = width,
		.rowBytes = CVPixelBufferGetBytesPerRow(pha),
	};
	vImage_Buffer raw = {.data = _incoming, .height = height, .width = width, .rowBytes = width};
	if (format == kCVPixelFormatType_OneComponent8) {
		vImageCopyBuffer(&src, &raw, 1, kvImageNoFlags);
	} else {
		vImage_Buffer tmp = {0};
		vImageBuffer_Init(&tmp, height, width, 32, kvImageNoFlags);
		vImageConvert_Planar16FtoPlanarF(&src, &tmp, kvImageNoFlags);
		vImageConvert_PlanarFtoPlanar8(&tmp, &raw, 255.0f, 0.0f, kvImageNoFlags);
		free(tmp.data);
	}
	CVPixelBufferUnlockBaseAddress(pha, kCVPixelBufferLock_ReadOnly);

	[self updateCoreFrom:_incoming width:width height:height];
	[self applyPlateTo:_incoming width:width height:height frame:bgra];
	[self applyChairTo:_incoming width:width height:height frame:bgra];

	// _incoming keeps the unblurred matte, so the flicker statistic measures
	// what the blur and the steadying take out together.
	vImage_Buffer unblurred = {.data = _incoming, .height = height, .width = width, .rowBytes = width};
	vImage_Buffer blurred = {.data = _edgeBlurred, .height = height, .width = width, .rowBytes = width};
	vImageTentConvolve_Planar8(&unblurred, &blurred, NULL, 0, 0, kEdgeBlur, kEdgeBlur, 0, kvImageEdgeExtend);

	// The motion grid is laid over the matte, so both must come from the
	// same model size.
	BOOL have_motion = _motionValid && _gridWidth == width / kMotionGridScale &&
			   _gridHeight == height / kMotionGridScale;
	if (_smoothValid && have_motion) {
		double raw_change = 0.0;
		for (size_t i = 0; i < count; i++)
			raw_change += abs((int)_incoming[i] - (int)_previous[i]);
		_statsRawChange += raw_change / (double)count;
		_statsChange += smooth_matte(_smoothed, _edgeBlurred, _incoming, _previous, width, height, _motion,
					     _gridWidth, _gridHeight, self.stability, &_statsEdge) /
				(double)count;
	} else {
		for (size_t i = 0; i < count; i++)
			_smoothed[i] = _edgeBlurred[i];
		_smoothValid = YES;
	}
	uint8_t *swap = _previous;
	_previous = _incoming;
	_incoming = swap;

	[_matteLock lock];
	if (!_slots[0] || _matteWidth != width || _matteHeight != height) {
		for (int s = 0; s < MATTE_SLOTS; s++) {
			free(_slots[s]);
			_slots[s] = malloc(count * 2);
			_slotTimestamps[s] = 0;
			_slotGenerations[s] = 0;
		}
		_matteWidth = width;
		_matteHeight = height;
	}
	_matteGeneration++;
	int slot = (int)(_matteGeneration % MATTE_SLOTS);
	// Each slot pixel holds the matte and, second, how strongly to remove the
	// chair's color spill there (only near where the chair was cut).
	uint8_t *dst = _slots[slot];
	const uint8_t *spill = _chairMaskValid ? _chairNear : NULL;
	for (size_t i = 0; i < count; i++) {
		dst[i * 2] = (uint8_t)(_smoothed[i] + 0.5f);
		dst[i * 2 + 1] = spill ? spill[i] : 0;
	}
	_slotTimestamps[slot] = timestamp;
	_slotGenerations[slot] = _matteGeneration;
	NSString *snapshot = nil;
	if (_snapshotPath && --_snapshotCountdown <= 0) {
		snapshot = _snapshotPath;
		_snapshotPath = nil;
	}
	[_matteLock unlock];
	self.latestMatteTimestamp = timestamp;

	if (snapshot) {
		[self writeMatteSnapshotTo:snapshot];
		[self logColorStatsOf:bgra];
	}

	if (--_autoColorCountdown <= 0) {
		_autoColorCountdown = kAutoColorEvery;
		if (self.autoColor) {
			[self updateAutoColorFrom:bgra];
		} else {
			for (int c = 0; c < 4; c++)
				_colorTarget[c] = 1.0f;
		}
	}
	[self moveColorToward:_colorTarget];
}

// Moves the color state a step toward target (gains, then lift). Values
// that are almost there snap to it, so "off" ends at exactly no change.
- (void)moveColorToward:(const float *)target
{
	[_colorLock lock];
	float *state[4] = {&_colorGains[0], &_colorGains[1], &_colorGains[2], &_colorLift};
	for (int c = 0; c < 4; c++) {
		*state[c] += (target[c] - *state[c]) * kAutoColorRate;
		if (fabsf(target[c] - *state[c]) < 0.002f)
			*state[c] = target[c];
	}
	[_colorLock unlock];
}

static float srgb_to_linear(int v)
{
	static float table[256];
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		for (int i = 0; i < 256; i++) {
			float c = (float)i / 255.0f;
			table[i] = c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f);
		}
	});
	return table[v];
}

// Measures the frame and sets new color targets: exposure from the
// background's highlights and white balance from plain background areas.
// Skin never sets the white balance, and sets the brightness only when
// almost no background is visible. Runs on the publish queue.
- (void)updateAutoColorFrom:(CVPixelBufferRef)bgra
{
	uint32_t width = (uint32_t)CVPixelBufferGetWidth(bgra);
	uint32_t height = (uint32_t)CVPixelBufferGetHeight(bgra);
	if (!_smoothed || width != _smoothWidth || height != _smoothHeight)
		return;

	double neutral[3] = {0};
	uint64_t neutral_n = 0, total = 0;
	// Brightness of unclipped pixels: background only, and the whole frame
	// as a fallback when the person fills almost all of it.
	uint64_t hist[256] = {0}, hist_all[256] = {0};
	CVPixelBufferLockBaseAddress(bgra, kCVPixelBufferLock_ReadOnly);
	const uint8_t *base = CVPixelBufferGetBaseAddress(bgra);
	size_t stride = CVPixelBufferGetBytesPerRow(bgra);
	for (uint32_t y = 0; y < height; y += 4) {
		for (uint32_t x = 0; x < width; x += 4) {
			const uint8_t *p = base + y * stride + (size_t)x * 4;
			int b = p[0], g = p[1], r = p[2];
			int luma = (54 * r + 183 * g + 19 * b) >> 8;
			int spread = MAX(r, MAX(g, b)) - MIN(r, MIN(g, b));
			total++;
			float m = _smoothed[(size_t)y * width + x];
			// Clipped pixels say nothing about exposure. The person is left
			// out, so skin never sets the brightness.
			if (MAX(r, MAX(g, b)) < 250) {
				hist_all[luma]++;
				if (m < 128.0f)
					hist[luma]++;
			}
			if (m <= 5.0f && luma > 60 && MAX(r, MAX(g, b)) < 245 && spread < 40) {
				neutral[0] += srgb_to_linear(r);
				neutral[1] += srgb_to_linear(g);
				neutral[2] += srgb_to_linear(b);
				neutral_n++;
			}
		}
	}
	CVPixelBufferUnlockBaseAddress(bgra, kCVPixelBufferLock_ReadOnly);

	// White balance: make the plain background neutral, keeping green fixed.
	float balance[3] = {1.0f, 1.0f, 1.0f};
	if (neutral_n >= (uint64_t)(kAutoBalanceMinShare * (float)total) && neutral[0] > 0 && neutral[2] > 0) {
		balance[0] = fminf(kAutoBalanceMax, fmaxf(kAutoBalanceMin, (float)(neutral[1] / neutral[0])));
		balance[2] = fminf(kAutoBalanceMax, fmaxf(kAutoBalanceMin, (float)(neutral[1] / neutral[2])));
	} else {
		[_colorLock lock];
		float gain_g = fmaxf(_colorGains[1], 0.01f);
		balance[0] = _colorGains[0] / gain_g;
		balance[2] = _colorGains[2] / gain_g;
		[_colorLock unlock];
	}

	// Exposure: bring the 98th-percentile brightness of the background to
	// the target.
	uint64_t counted = 0, seen = 0;
	for (int v = 0; v < 256; v++)
		counted += hist[v];
	if (counted < (uint64_t)(kAutoBackgroundMinShare * (float)total)) {
		memcpy(hist, hist_all, sizeof(hist));
		counted = 0;
		for (int v = 0; v < 256; v++)
			counted += hist[v];
	}
	int highlight = 255;
	for (int v = 0; v < 256 && counted; v++) {
		seen += hist[v];
		if (seen * 100 >= counted * 98) {
			highlight = v;
			break;
		}
	}
	float highlight_linear = fmaxf(srgb_to_linear(highlight), 0.01f);
	float gain = fminf(kAutoGainMax, fmaxf(kAutoGainMin, kAutoHighlightTarget / highlight_linear));

	_colorTarget[0] = gain * balance[0];
	_colorTarget[1] = gain * balance[1];
	_colorTarget[2] = gain * balance[2];
	_colorTarget[3] = kAutoColorLift;
}

- (void)getColorState:(float *)state
{
	[_colorLock lock];
	state[0] = _colorGains[0];
	state[1] = _colorGains[1];
	state[2] = _colorGains[2];
	state[3] = _colorLift;
	[_colorLock unlock];
}

- (void)setColorState:(const float *)state
{
	[_colorLock lock];
	_colorGains[0] = state[0];
	_colorGains[1] = state[1];
	_colorGains[2] = state[2];
	_colorLift = state[3];
	// The publish queue is still idle, so its targets can be set here too.
	for (int c = 0; c < 4; c++)
		_colorTarget[c] = state[c];
	[_colorLock unlock];
}

// Logs color statistics of the camera frame that the snapshot came from, for
// tuning color correction without saving the picture: the mean color of
// skin, of bright plain background and of dark plain clothing, and the spread
// of brightness. Runs on the publish queue.
- (void)logColorStatsOf:(CVPixelBufferRef)bgra
{
	uint32_t width = (uint32_t)CVPixelBufferGetWidth(bgra);
	uint32_t height = (uint32_t)CVPixelBufferGetHeight(bgra);
	if (!_smoothed || width != _smoothWidth || height != _smoothHeight)
		return;

	double skin[3] = {0}, wall[3] = {0}, dark[3] = {0};
	uint64_t skin_n = 0, wall_n = 0, dark_n = 0, total = 0;
	uint64_t hist[256] = {0};
	CVPixelBufferLockBaseAddress(bgra, kCVPixelBufferLock_ReadOnly);
	const uint8_t *base = CVPixelBufferGetBaseAddress(bgra);
	size_t stride = CVPixelBufferGetBytesPerRow(bgra);
	for (uint32_t y = 0; y < height; y += 2) {
		for (uint32_t x = 0; x < width; x += 2) {
			const uint8_t *p = base + y * stride + (size_t)x * 4;
			int b = p[0], g = p[1], r = p[2];
			int luma = (54 * r + 183 * g + 19 * b) >> 8;
			int spread = MAX(r, MAX(g, b)) - MIN(r, MIN(g, b));
			float m = _smoothed[(size_t)y * width + x];
			hist[luma]++;
			total++;
			if (m >= 250.0f) {
				float warmth = r > 0 ? (float)(r - g) / (float)r : 0.0f;
				if (r > g && g > b && r > 60 && warmth > 0.12f && warmth < 0.45f) {
					skin[0] += r, skin[1] += g, skin[2] += b;
					skin_n++;
				} else if (luma < 110 && spread < 20) {
					dark[0] += r, dark[1] += g, dark[2] += b;
					dark_n++;
				}
			} else if (m <= 5.0f && luma > 140 && spread < 30) {
				wall[0] += r, wall[1] += g, wall[2] += b;
				wall_n++;
			}
		}
	}
	CVPixelBufferUnlockBaseAddress(bgra, kCVPixelBufferLock_ReadOnly);

	int pct[3] = {0}, wanted[3] = {1, 50, 99}, k = 0;
	uint64_t seen = 0;
	for (int v = 0; v < 256 && k < 3; v++) {
		seen += hist[v];
		while (k < 3 && seen * 100 >= total * (uint64_t)wanted[k])
			pct[k++] = v;
	}
#define MEAN(a, n, i) ((n) ? (a)[i] / (double)(n) : 0.0)
	blog_rvm(LOG_INFO,
		 "Camera color: skin %.0f %.0f %.0f, bright background %.0f %.0f %.0f, dark clothing %.0f %.0f %.0f "
		 "(R G B), brightness p1 %d p50 %d p99 %d",
		 MEAN(skin, skin_n, 0), MEAN(skin, skin_n, 1), MEAN(skin, skin_n, 2), MEAN(wall, wall_n, 0),
		 MEAN(wall, wall_n, 1), MEAN(wall, wall_n, 2), MEAN(dark, dark_n, 0), MEAN(dark, dark_n, 1),
		 MEAN(dark, dark_n, 2), pct[0], pct[1], pct[2]);
#undef MEAN
}

- (BOOL)copyMatteFor:(uint64_t)timestamp
	  generation:(uint64_t *)generation
		into:(uint8_t **)dst
	       width:(uint32_t *)width
	      height:(uint32_t *)height
{
	BOOL copied = NO;
	[_matteLock lock];
	int best = -1;
	int newest = -1;
	for (int s = 0; s < MATTE_SLOTS; s++) {
		if (!_slotGenerations[s])
			continue;
		if (newest < 0 || _slotGenerations[s] > _slotGenerations[newest])
			newest = s;
		if (_slotTimestamps[s] <= timestamp && (best < 0 || _slotGenerations[s] > _slotGenerations[best]))
			best = s;
	}
	if (best < 0)
		best = newest;
	if (best >= 0 && _slotGenerations[best] != *generation) {
		size_t size = (size_t)_matteWidth * _matteHeight * 2;
		if (*width != _matteWidth || *height != _matteHeight) {
			*dst = brealloc(*dst, size);
			*width = _matteWidth;
			*height = _matteHeight;
		}
		memcpy(*dst, _slots[best], size);
		*generation = _slotGenerations[best];
		copied = YES;
	}
	[_matteLock unlock];
	return copied;
}

- (void)requestSnapshotTo:(NSString *)path
{
	[_matteLock lock];
	_snapshotPath = [path copy];
	_snapshotCountdown = kSnapshotDelayMattes;
	[_matteLock unlock];
}

static void write_gray_png(const uint8_t *data, uint32_t width, uint32_t height, NSString *path)
{
	CGColorSpaceRef gray = CGColorSpaceCreateDeviceGray();
	CGContextRef ctx = CGBitmapContextCreate((void *)data, width, height, 8, width, gray,
						 (CGBitmapInfo)kCGImageAlphaNone);
	CGImageRef image = ctx ? CGBitmapContextCreateImage(ctx) : NULL;
	if (image) {
		CGImageDestinationRef dest = CGImageDestinationCreateWithURL(
			(__bridge CFURLRef)[NSURL fileURLWithPath:path], (__bridge CFStringRef)UTTypePNG.identifier,
			1, NULL);
		if (dest) {
			CGImageDestinationAddImage(dest, image, NULL);
			CGImageDestinationFinalize(dest);
			CFRelease(dest);
			blog_rvm(LOG_INFO, "Wrote snapshot to %s", path.UTF8String);
		}
		CGImageRelease(image);
	}
	if (ctx)
		CGContextRelease(ctx);
	CGColorSpaceRelease(gray);
}

// Writes the current matte as a grayscale PNG, for checking edge quality
// without capturing the camera image itself. When chair removal is on, what
// it cut is written next to it with a "-chair" suffix.
- (void)writeMatteSnapshotTo:(NSString *)path
{
	// Copy the matte channel under the lock and encode outside it, so the
	// video thread never waits for PNG encoding.
	[_matteLock lock];
	uint32_t width = _matteWidth, height = _matteHeight;
	size_t count = (size_t)width * height;
	uint8_t *matte = malloc(count);
	const uint8_t *slot = _slots[_matteGeneration % MATTE_SLOTS];
	for (size_t i = 0; i < count; i++)
		matte[i] = slot[i * 2];
	[_matteLock unlock];
	write_gray_png(matte, width, height, path);
	free(matte);
	if (_chairMaskValid) {
		NSString *base = [path stringByDeletingPathExtension];
		write_gray_png(_chairCut, (uint32_t)_modelWidth, (uint32_t)_modelHeight,
			       [base stringByAppendingString:@"-chair.png"]);
		// The chair score of every grid cell, for tuning the color test.
		// It holds no picture detail beyond redness.
		size_t grid_count = (size_t)_gridWidth * _gridHeight;
		uint8_t *score = malloc(grid_count);
		for (size_t i = 0; i < grid_count; i++) {
			const uint8_t *p = _gridBGRA + i * 4;
			score[i] = (uint8_t)(chair_score(p[2], p[1], p[0]) * 255.0f + 0.5f);
		}
		write_gray_png(score, _gridWidth, _gridHeight, [base stringByAppendingString:@"-chairscore.png"]);
		free(score);
	}
}

- (void)recordTiming:(double)ms
{
	_statsTotalMs += ms;
	_statsMaxMs = MAX(_statsMaxMs, ms);
	_statsCount++;

	double now = CFAbsoluteTimeGetCurrent();
	double elapsed = now - _statsStart;
	if (elapsed >= kStatsIntervalSeconds) {
		blog_rvm(LOG_INFO,
			 "Matte updates: %.1f per second, inference avg %.1f ms, max %.1f ms, frames skipped %llu, "
			 "flicker %.3f raw / %.3f shown, still-edge flicker %.2f raw / %.2f shown",
			 _statsCount / elapsed, _statsTotalMs / (double)_statsCount, _statsMaxMs,
			 (unsigned long long)_statsSkipped, _statsRawChange / (double)_statsCount,
			 _statsChange / (double)_statsCount, _statsEdge.raw / (double)MAX(_statsEdge.pixels, 1),
			 _statsEdge.shown / (double)MAX(_statsEdge.pixels, 1));
		if (self.autoColor) {
			float gains[4];
			[self getColorState:gains];
			blog_rvm(LOG_INFO, "Automatic color gains: red %.2f, green %.2f, blue %.2f", gains[0],
				 gains[1], gains[2]);
		}
		_statsStart = now;
		_statsRawChange = 0;
		_statsChange = 0;
		_statsEdge = (struct edge_flicker){0};
		_statsTotalMs = 0;
		_statsMaxMs = 0;
		_statsCount = 0;
		_statsSkipped = 0;
	}
}

- (void)drain
{
	self.canceled = YES;
	dispatch_sync(_queue, ^{
	});
	dispatch_sync(_postQueue, ^{
	});
}

@end

#pragma mark - OBS filter

// Loads engines in the background for one filter, so changing the model
// never stalls the OBS video thread. Loads run one at a time on a serial
// queue, and a load whose request has been superseded, or whose filter is
// gone, is skipped or dropped instead of swapped in. ready and closed are
// only read and changed under @synchronized(loader).
@interface RVMEngineLoader : NSObject
@property (atomic) uint64_t request;
@property (nonatomic, strong) RVMEngine *ready;
@property (nonatomic) BOOL closed;
@property (nonatomic, readonly) dispatch_queue_t queue;
@property (nonatomic, readonly) dispatch_group_t group;
@end

@implementation RVMEngineLoader
- (instancetype)init
{
	if ((self = [super init])) {
		_queue = dispatch_queue_create("rvm-matting.load", DISPATCH_QUEUE_SERIAL);
		_group = dispatch_group_create();
	}
	return self;
}
@end

struct rvm_filter {
	obs_source_t *context;
	gs_effect_t *effect;
	gs_eparam_t *param_matte;
	gs_eparam_t *param_low;
	gs_eparam_t *param_high;
	gs_eparam_t *param_gain;
	gs_eparam_t *param_lift;

	void *engine; // RVMEngine, retained via CFBridgingRetain
	void *loader; // RVMEngineLoader, retained via CFBridgingRetain
	char *model_name;
	long long compute_units;
	char *snapshot_name; // a requested debug snapshot, waiting for an engine
	bool loading;        // a model load has been started and not yet adopted

	gs_texture_t *matte_tex;
	uint8_t *matte_copy;
	uint32_t matte_width;
	uint32_t matte_height;
	uint64_t matte_generation;

	float alpha_low;
	float alpha_high;
	float stability;

	// "Line up cutout with video": frames wait here until their own matte is
	// ready. OBS calls filter_video, update and video_render on its video
	// thread, so these need no lock.
	bool sync;
	struct obs_source_frame *held[MAX_HELD_FRAMES];
	size_t held_count;
	uint64_t last_timestamp;
	// Timestamp of the frame being shown; UINT64_MAX shows the newest matte.
	uint64_t shown_timestamp;

	// Set by the property buttons on the UI thread and passed to the engine
	// on the video thread.
	volatile bool plate_capture_requested;
	volatile bool plate_forget_requested;
	bool use_plate;
	bool remove_chair;
	bool auto_color;
};

static void release_held_frames(struct rvm_filter *f, obs_source_t *parent)
{
	for (size_t i = 0; i < f->held_count; i++)
		obs_source_release_frame(parent, f->held[i]);
	f->held_count = 0;
}

static RVMEngine *engine_of(struct rvm_filter *f)
{
	return (__bridge RVMEngine *)f->engine;
}

static RVMEngineLoader *loader_of(struct rvm_filter *f)
{
	return (__bridge RVMEngineLoader *)f->loader;
}

// Stops the current engine. On the video thread the engine is drained in
// the background, so a running inference never stalls the video.
static void retire_engine(struct rvm_filter *f, bool wait)
{
	if (!f->engine)
		return;
	RVMEngine *engine = CFBridgingRelease(f->engine);
	f->engine = NULL;
	if (wait) {
		[engine drain];
	} else {
		dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
			[engine drain];
		});
	}
}

// Passes the filter's current settings, and any waiting snapshot request, to
// the engine.
static void apply_engine_settings(struct rvm_filter *f)
{
	if (!f->engine)
		return;
	RVMEngine *engine = engine_of(f);
	engine.stability = f->stability;
	engine.usePlate = f->use_plate;
	engine.removeChair = f->remove_chair;
	engine.autoColor = f->auto_color;
	// A snapshot waits for a model that is loading, so a switch does not
	// retire the engine before it writes the snapshot.
	if (f->snapshot_name && !f->loading) {
		char *dir = obs_module_config_path("");
		if (dir) {
			[engine requestSnapshotTo:[@(dir) stringByAppendingPathComponent:@(f->snapshot_name)]];
			bfree(dir);
		}
		bfree(f->snapshot_name);
		f->snapshot_name = NULL;
	}
}

// Swaps in an engine that finished loading in the background. Runs on the
// video thread.
static void adopt_ready_engine(struct rvm_filter *f)
{
	RVMEngineLoader *loader = loader_of(f);
	RVMEngine *ready = nil;
	@synchronized(loader) {
		ready = loader.ready;
		loader.ready = nil;
	}
	if (!ready)
		return;
	if (loader.request == ready.request)
		f->loading = false;
	if (f->engine) {
		float state[4];
		[engine_of(f) getColorState:state];
		[ready setColorState:state];
	}
	retire_engine(f, false);
	f->engine = (void *)CFBridgingRetain(ready);
	f->matte_generation = 0;
	apply_engine_settings(f);
}

// Starts loading the model in the background. The current engine keeps
// running until the new one is ready.
static void load_engine(struct rvm_filter *f, const char *model, long long units)
{
	char *path = obs_module_file(model);
	if (!path) {
		blog_rvm(LOG_ERROR, "Model %s is missing from the plugin bundle", model);
		return;
	}
	char *config_dir = obs_module_config_path("");
	if (config_dir)
		os_mkdirs(config_dir);
	bfree(config_dir);
	char *plate_path = obs_module_config_path("background-photo.png");

	NSString *model_path = @(path);
	NSString *plate = plate_path ? @(plate_path) : nil;
	bfree(path);
	bfree(plate_path);

	RVMEngineLoader *loader = loader_of(f);
	uint64_t request = loader.request + 1;
	loader.request = request;
	f->loading = true;
	dispatch_group_async(loader.group, loader.queue, ^{
		@synchronized(loader) {
			if (loader.closed || loader.request != request)
				return;
		}
		RVMEngine *engine = [[RVMEngine alloc] initWithModelPath:model_path
							     computeUnits:(MLComputeUnits)units
								platePath:plate];
		if (!engine)
			return;
		engine.request = request;
		RVMEngine *unused = engine;
		@synchronized(loader) {
			if (!loader.closed && loader.request == request) {
				unused = loader.ready;
				loader.ready = engine;
			}
		}
		[unused drain];
	});
}

static const char *rvm_get_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("PersonMatting");
}

static void rvm_update(void *data, obs_data_t *settings)
{
	struct rvm_filter *f = data;

	f->alpha_low = (float)obs_data_get_double(settings, "alpha_low");
	f->alpha_high = (float)obs_data_get_double(settings, "alpha_high");
	if (f->alpha_high <= f->alpha_low + 0.01f)
		f->alpha_high = f->alpha_low + 0.01f;
	f->stability = (float)obs_data_get_double(settings, "edge_stability");
	f->sync = obs_data_get_bool(settings, "sync_video");
	f->use_plate = obs_data_get_bool(settings, "use_plate");
	f->remove_chair = obs_data_get_bool(settings, "remove_chair");
	f->auto_color = obs_data_get_bool(settings, "auto_color");

	// Only the bundled models and the listed compute units can be used, so a
	// scene collection cannot point the filter at another file.
	const char *model = obs_data_get_string(settings, "model");
	if (strcmp(model, kModelAccurate.UTF8String) != 0 && strcmp(model, kModelFast.UTF8String) != 0) {
		blog_rvm(LOG_WARNING, "Unknown model setting; using the high accuracy model");
		model = kModelAccurate.UTF8String;
	}
	long long units = obs_data_get_int(settings, "compute_units");
	if (units != MLComputeUnitsCPUAndNeuralEngine && units != MLComputeUnitsCPUAndGPU && units != MLComputeUnitsAll)
		units = MLComputeUnitsCPUAndNeuralEngine;
	if (!f->model_name || strcmp(f->model_name, model) != 0 || units != f->compute_units) {
		bfree(f->model_name);
		f->model_name = bstrdup(model);
		f->compute_units = units;
		load_engine(f, model, units);
	}

	// Hidden setting used to check edge quality: writes the next matte as a
	// PNG with this file name into the plugin's config folder, then clears
	// itself. Only a plain .png file name is accepted, and never the name of
	// the saved background photo.
	const char *snapshot = obs_data_get_string(settings, "debug_snapshot");
	if (snapshot && *snapshot) {
		NSString *name = @(snapshot);
		if ([name isEqualToString:name.lastPathComponent] && ![name hasPrefix:@"."] &&
		    [name.pathExtension.lowercaseString isEqualToString:@"png"] &&
		    ![name isEqualToString:@"background-photo.png"]) {
			bfree(f->snapshot_name);
			f->snapshot_name = bstrdup(snapshot);
		} else {
			blog_rvm(LOG_WARNING, "Ignoring debug_snapshot: it must be a plain .png file name");
		}
		obs_data_set_string(settings, "debug_snapshot", "");
	}
	apply_engine_settings(f);
}

static void *rvm_create(obs_data_t *settings, obs_source_t *context)
{
	struct rvm_filter *f = bzalloc(sizeof(*f));
	f->context = context;
	f->shown_timestamp = UINT64_MAX;
	f->loader = (void *)CFBridgingRetain([RVMEngineLoader new]);

	char *effect_path = obs_module_file("rvm-matting.effect");
	obs_enter_graphics();
	f->effect = gs_effect_create_from_file(effect_path, NULL);
	if (f->effect) {
		f->param_matte = gs_effect_get_param_by_name(f->effect, "matte");
		f->param_low = gs_effect_get_param_by_name(f->effect, "alpha_low");
		f->param_high = gs_effect_get_param_by_name(f->effect, "alpha_high");
		f->param_gain = gs_effect_get_param_by_name(f->effect, "color_gain");
		f->param_lift = gs_effect_get_param_by_name(f->effect, "color_lift");
	}
	obs_leave_graphics();
	bfree(effect_path);

	if (!f->effect) {
		blog_rvm(LOG_ERROR, "Could not load rvm-matting.effect");
		CFBridgingRelease(f->loader);
		bfree(f);
		return NULL;
	}

	rvm_update(f, settings);
	return f;
}

static void rvm_destroy(void *data)
{
	struct rvm_filter *f = data;

	// Stop further loads and wait for one in flight, so no plugin code runs
	// after the filter (or, at shutdown, the plugin) is gone. Once closed is
	// set, no load can store an engine in ready.
	RVMEngineLoader *loader = CFBridgingRelease(f->loader);
	@synchronized(loader) {
		loader.closed = YES;
	}
	dispatch_group_wait(loader.group, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC));
	RVMEngine *unused = nil;
	@synchronized(loader) {
		unused = loader.ready;
		loader.ready = nil;
	}
	[unused drain];
	retire_engine(f, true);

	obs_enter_graphics();
	gs_effect_destroy(f->effect);
	gs_texture_destroy(f->matte_tex);
	obs_leave_graphics();

	bfree(f->matte_copy);
	bfree(f->model_name);
	bfree(f->snapshot_name);
	bfree(f);
}

// Frames still held when the filter is removed go back to the camera source,
// as in OBS's own async delay filter.
static void rvm_filter_remove(void *data, obs_source_t *parent)
{
	release_held_frames(data, parent);
}

static struct obs_source_frame *rvm_filter_video(void *data, struct obs_source_frame *frame)
{
	struct rvm_filter *f = data;
	obs_source_t *parent = obs_filter_get_parent(f->context);
	adopt_ready_engine(f);
	bool accepted = false;
	if (f->engine) {
		if (os_atomic_set_bool(&f->plate_capture_requested, false))
			[engine_of(f) requestPlateCaptureIn:kPlateCaptureDelaySeconds];
		if (os_atomic_set_bool(&f->plate_forget_requested, false))
			[engine_of(f) requestPlateForget];
		accepted = [engine_of(f) acceptsFrame:frame];
		[engine_of(f) submitFrame:frame];
	}

	// Frames go straight through when lining up is off, and also when no
	// matte can be expected: no engine yet, a frame format the engine cannot
	// read, or no matte for a while (for example after inference failures).
	uint64_t ready = f->engine ? engine_of(f).latestMatteTimestamp : 0;
	bool matte_stalled = !ready || (frame->timestamp > ready && frame->timestamp - ready > kSyncGiveUpNs);
	if (!f->sync || !accepted || matte_stalled) {
		release_held_frames(f, parent);
		f->shown_timestamp = UINT64_MAX;
		return frame;
	}

	// Start over after a pause or a timestamp jump, so stale frames are
	// never shown.
	if (f->held_count &&
	    (frame->timestamp < f->last_timestamp || frame->timestamp - f->last_timestamp > 1000000000ULL))
		release_held_frames(f, parent);
	f->last_timestamp = frame->timestamp;
	f->held[f->held_count++] = frame;

	// Show the newest frame that has a matte; older ones are dropped.
	ssize_t pick = -1;
	for (size_t i = 0; i < f->held_count; i++) {
		if (f->held[i]->timestamp <= ready)
			pick = (ssize_t)i;
	}
	if (pick < 0) {
		if (f->held_count < MAX_HELD_FRAMES)
			return NULL;
		pick = 0; // the model has fallen behind; keep the video moving
	}

	for (ssize_t i = 0; i < pick; i++)
		obs_source_release_frame(parent, f->held[i]);
	struct obs_source_frame *out = f->held[pick];
	size_t rest = f->held_count - (size_t)pick - 1;
	memmove(f->held, f->held + pick + 1, rest * sizeof(f->held[0]));
	f->held_count = rest;
	f->shown_timestamp = out->timestamp;
	return out;
}

static void rvm_video_render(void *data, gs_effect_t *unused)
{
	UNUSED_PARAMETER(unused);
	struct rvm_filter *f = data;

	if (f->engine) {
		uint32_t old_width = f->matte_width, old_height = f->matte_height;
		if ([engine_of(f) copyMatteFor:f->shown_timestamp
				    generation:&f->matte_generation
					  into:&f->matte_copy
					 width:&f->matte_width
					height:&f->matte_height]) {
			if (!f->matte_tex || old_width != f->matte_width || old_height != f->matte_height) {
				gs_texture_destroy(f->matte_tex);
				f->matte_tex = gs_texture_create(f->matte_width, f->matte_height, GS_R8G8, 1, NULL,
								 GS_DYNAMIC);
			}
			if (f->matte_tex)
				gs_texture_set_image(f->matte_tex, f->matte_copy, f->matte_width * 2, false);
		}
	}

	if (!f->matte_tex || !f->engine) {
		obs_source_skip_video_filter(f->context);
		return;
	}

	// Same color-space handling and premultiplied blending as OBS's own
	// chroma key filter, so this filter can sit anywhere in a filter chain.
	const enum gs_color_space preferred_spaces[] = {
		GS_CS_SRGB,
		GS_CS_SRGB_16F,
		GS_CS_709_EXTENDED,
	};
	const enum gs_color_space space = obs_source_get_color_space(obs_filter_get_target(f->context),
								     OBS_COUNTOF(preferred_spaces), preferred_spaces);
	if (!obs_source_process_filter_begin_with_color_space(f->context, gs_get_format_from_space(space), space,
							      OBS_ALLOW_DIRECT_RENDERING))
		return;

	gs_effect_set_texture(f->param_matte, f->matte_tex);
	gs_effect_set_float(f->param_low, f->alpha_low);
	gs_effect_set_float(f->param_high, f->alpha_high);
	// The shader's color math assumes sRGB-encoded 8-bit input, so wide-gamut
	// and HDR sources are left unchanged.
	float state[4] = {1.0f, 1.0f, 1.0f, 1.0f};
	if (space == GS_CS_SRGB)
		[engine_of(f) getColorState:state];
	struct vec3 gain;
	vec3_set(&gain, state[0], state[1], state[2]);
	gs_effect_set_vec3(f->param_gain, &gain);
	gs_effect_set_float(f->param_lift, state[3]);

	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_INVSRCALPHA);
	obs_source_process_filter_end(f->context, f->effect, 0, 0);
	gs_blend_state_pop();
}

static void rvm_defaults(obs_data_t *settings)
{
	obs_data_set_default_string(settings, "model", kModelAccurate.UTF8String);
	obs_data_set_default_int(settings, "compute_units", MLComputeUnitsCPUAndNeuralEngine);
	obs_data_set_default_double(settings, "alpha_low", 0.25);
	obs_data_set_default_double(settings, "alpha_high", 0.75);
	obs_data_set_default_double(settings, "edge_stability", 0.85);
	obs_data_set_default_bool(settings, "sync_video", true);
	obs_data_set_default_bool(settings, "use_plate", true);
	obs_data_set_default_bool(settings, "remove_chair", false);
	obs_data_set_default_bool(settings, "auto_color", false);
}

static bool rvm_capture_plate_clicked(obs_properties_t *props, obs_property_t *property, void *data)
{
	UNUSED_PARAMETER(props);
	UNUSED_PARAMETER(property);
	struct rvm_filter *f = data;
	os_atomic_set_bool(&f->plate_capture_requested, true);
	return false;
}

static bool rvm_forget_plate_clicked(obs_properties_t *props, obs_property_t *property, void *data)
{
	UNUSED_PARAMETER(props);
	UNUSED_PARAMETER(property);
	struct rvm_filter *f = data;
	os_atomic_set_bool(&f->plate_forget_requested, true);
	return false;
}

static obs_properties_t *rvm_properties(void *data)
{
	obs_properties_t *props = obs_properties_create();

	obs_property_t *model = obs_properties_add_list(props, "model", obs_module_text("Model"),
							OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(model, obs_module_text("Model.Accurate"), kModelAccurate.UTF8String);
	obs_property_list_add_string(model, obs_module_text("Model.Fast"), kModelFast.UTF8String);

	obs_property_t *units = obs_properties_add_list(props, "compute_units", obs_module_text("ComputeUnits"),
							OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(units, obs_module_text("ComputeUnits.NeuralEngine"),
				  MLComputeUnitsCPUAndNeuralEngine);
	obs_property_list_add_int(units, obs_module_text("ComputeUnits.GPU"), MLComputeUnitsCPUAndGPU);
	obs_property_list_add_int(units, obs_module_text("ComputeUnits.All"), MLComputeUnitsAll);

	obs_properties_add_float_slider(props, "alpha_low", obs_module_text("AlphaLow"), 0.0, 0.5, 0.01);
	obs_properties_add_float_slider(props, "alpha_high", obs_module_text("AlphaHigh"), 0.5, 1.0, 0.01);
	obs_property_t *stability = obs_properties_add_float_slider(props, "edge_stability",
								     obs_module_text("EdgeStability"), 0.0, 1.0, 0.05);
	obs_property_set_long_description(stability, obs_module_text("EdgeStability.Description"));
	obs_property_t *sync = obs_properties_add_bool(props, "sync_video", obs_module_text("SyncVideo"));
	obs_property_set_long_description(sync, obs_module_text("SyncVideo.Description"));

	obs_property_t *auto_color = obs_properties_add_bool(props, "auto_color", obs_module_text("AutoColor"));
	obs_property_set_long_description(auto_color, obs_module_text("AutoColor.Description"));

	obs_property_t *remove_chair = obs_properties_add_bool(props, "remove_chair", obs_module_text("RemoveChair"));
	obs_property_set_long_description(remove_chair, obs_module_text("RemoveChair.Description"));

	obs_property_t *use_plate = obs_properties_add_bool(props, "use_plate", obs_module_text("UsePlate"));
	obs_property_set_long_description(use_plate, obs_module_text("UsePlate.Description"));
	obs_properties_add_button2(props, "capture_plate", obs_module_text("CapturePlate"), rvm_capture_plate_clicked,
				   data);
	obs_properties_add_button2(props, "forget_plate", obs_module_text("ForgetPlate"), rvm_forget_plate_clicked,
				   data);
	return props;
}

bool obs_module_load(void)
{
	static struct obs_source_info info = {
		.id = "rvm_person_matting",
		.type = OBS_SOURCE_TYPE_FILTER,
		.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_ASYNC,
		.get_name = rvm_get_name,
		.create = rvm_create,
		.destroy = rvm_destroy,
		.update = rvm_update,
		.get_defaults = rvm_defaults,
		.get_properties = rvm_properties,
		.filter_video = rvm_filter_video,
		.filter_remove = rvm_filter_remove,
		.video_render = rvm_video_render,
	};
	obs_register_source(&info);
	blog_rvm(LOG_INFO, "Plugin loaded (version %s)", PLUGIN_VERSION);
	return true;
}
