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
// video_render cleans the edge colors, lights the person to suit the chosen
// room background, and multiplies the source's alpha by a matte on the GPU.
//
// With "line up cutout with video" on, filter_video holds each frame until
// its own matte is ready, so the video is delayed by one or two frames and
// moving hands stay inside the cutout. With it off, frames pass straight
// through and the newest matte trails them by one inference.

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#import <VideoToolbox/VideoToolbox.h>
#import <Accelerate/Accelerate.h>
#import <AppKit/AppKit.h>

#include <obs-module.h>
#include <util/platform.h>
#include <util/threading.h>
#include <graphics/vec2.h>
#include <graphics/vec3.h>
#include <graphics/vec4.h>

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
// The model wavers by a few levels. A change that averages kSteadyFollowHigh
// levels or more over three by three grid cells is a real change, so it shows at
// once even where the picture looks still. Dark hair moving over dark headphones
// barely changes the picture, and without this a gap that has closed stays
// visible as a square patch of grid cells. Changes between the two levels are
// eased in. The average matters: a crisp edge that wobbles by one pixel changes
// single pixels by up to 85 levels, but its average stays near 21, so it is
// still steadied.
static const float kSteadyFollowLow = 32.0f;
static const float kSteadyFollowHigh = 64.0f;
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
// largest color difference of any channel is at most kPlateMatchLow levels
// the pixel is background and is cut out; from kPlateMatchHigh up it is left
// to the model. Both are raised by kPlateNoiseScale times each pixel's camera
// noise. The limits are tight so that skin cannot pass for a wall.
static const float kPlateMatchLow = 5.0f;
static const float kPlateMatchHigh = 14.0f;
static const float kPlateNoiseScale = 1.0f;
// Apart from thin gaps, the photo only cuts pixels the model is unsure about,
// below this matte value. Otherwise any skin close to the wall color would go.
static const uint8_t kPlateSureMatte = 250;
// The photo averages this many frames, which also measures each pixel's noise.
// The noise is saved beside the photo; a photo without it uses kPlateNoiseDefault.
static const int kPlateCaptureFrames = 30;
static const uint8_t kPlateNoiseDefault = 1;
// Camera exposure shifts when someone sits down, so the frame is matched to
// the photo first: each channel's gain comes from pixels the model calls
// background, sampled every kPlateGainStep pixels, within kPlateGainMin..Max.
static const size_t kPlateGainStep = 8;
static const float kPlateGainMin = 0.7f;
static const float kPlateGainMax = 1.4f;
static const float kPlateGainRate = 0.1f;
// Thin gaps, such as the room showing between a headband and a temple, may be
// cut even inside the protected middle of the person. A pixel qualifies if it
// matches the photo within kPlateGapMatch levels, belongs to a match narrower
// than kPlateGapWidth pixels (odd), survives a kPlateGapSpeck window (odd),
// lies within kPlateGapReach motion-grid cells (odd, so about half that
// across) of the person's edge, and is in the upper half of the person. Those
// last two keep the gap rule on the head and away from patterned clothes.
static const float kPlateGapMatch = 6.0f;
// A gap must look brighter than this in the photo, in levels of its largest
// color channel. Dark headphones cannot be told from a dark chair behind them,
// so only bright parts of the room can open a gap.
static const uint8_t kPlateGapMinLevel = 96;
static const vImagePixelCount kPlateGapWidth = 41;
static const vImagePixelCount kPlateGapSpeck = 3;
static const vImagePixelCount kPlateGapReach = 33;
// The color difference is averaged over this many pixels (odd) to ignore
// camera noise.
static const vImagePixelCount kPlateBlur = 5;
// The solid middle of the person, shrunk by this many motion-grid cells
// (odd), is not cut by the photo, apart from thin gaps, so clothes that match
// what is behind them stay whole.
static const vImagePixelCount kCoreShrink = 9;
// Gaps in the solid middle narrower than this many cells (odd) are filled
// first. Without this, one speckled cell in the matte would remove a whole
// kCoreShrink square of protection.
static const vImagePixelCount kCoreGapFill = 3;
// Where the model is sure it sees background and nothing moves, the photo
// creeps toward the live picture by one level every kPlateAdaptEvery mattes,
// following slow lighting drift. Differences of kPlateAdaptLimit levels or more
// are ignored, so a moment of model doubt can never teach the photo a person.
static const int kPlateAdaptEvery = 8;
static const int kPlateAdaptLimit = 24;
// The photo is taken this long after the capture button is clicked.
static const double kPlateCaptureDelaySeconds = 5.0;

// "Blend into the room": the room background is rendered small and halved again
// and again, which blurs it for free. The shader takes edge light from
// kRoomWrapMap, the room's light direction from kRoomLightMap and its overall
// color from kRoomMeanMap, the last one, which is a single pixel.
static const struct {
	uint32_t width;
	uint32_t height;
} kRoomMaps[] = {{512, 288}, {256, 144}, {128, 72}, {64, 36}, {32, 18}, {16, 9}, {8, 4}, {4, 2}, {2, 1}, {1, 1}};
#define ROOM_MAP_COUNT (sizeof(kRoomMaps) / sizeof(kRoomMaps[0]))
static const size_t kRoomWrapMap = 3;
static const size_t kRoomLightMap = 6;
static const size_t kRoomMeanMap = ROOM_MAP_COUNT - 1;
// This is how far around a pixel the shader looks for the edge of the person,
// as a share of the picture height (13.5 pixels at 1080p).
static const float kRoomEdgeReach = 1.0f / 80.0f;
// A room source that is missing, usually because it has not loaded yet, is
// looked up again after this long.
static const uint64_t kRoomLookupNs = 1000000000ULL;

// This is how far inside the person "Clean edge colors" looks for a clean color,
// as a share of the picture height (4.5 pixels at 1080p).
static const float kEdgeReach = 1.0f / 240.0f;
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
// White balance moves plain background areas (walls) part of the way toward
// neutral: kAutoBalanceStrength of the full correction, because many walls
// really are warm, and a full correction turns skin cool and pale. It needs
// at least this share of the measured pixels, and each channel's correction
// stays within kAutoBalanceMin..Max.
static const float kAutoBalanceMinShare = 0.005f;
static const float kAutoBalanceMin = 0.75f;
static const float kAutoBalanceMax = 1.33f;
static const float kAutoBalanceStrength = 0.35f;
// After the gain, mid-tones are lifted with this power in linear light, much
// like a phone camera's tone curve.
static const float kAutoColorLift = 0.92f;
// Exposure is measured on the background when it makes up at least this
// share of the frame, and on the whole frame otherwise.
static const float kAutoBackgroundMinShare = 0.05f;

// Low light: the model cuts better on a bright picture, so a dark frame is
// lifted for the model only. Viewers keep the normal picture. The frame's mean
// linear brightness, sampled every kLiftSampleStep pixels, is brought up to
// kLiftTarget, by at most kLiftMax, moving kLiftRate of the way each frame.
static const float kLiftTarget = 0.18f;
static const float kLiftMax = 8.0f;
static const float kLiftRate = 0.05f;
static const size_t kLiftSampleStep = 16;
// Gains closer to 1 than this leave the frame alone.
static const float kLiftMinimum = 1.03f;

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
// Starts averaging frames for the empty-background photo after this many
// seconds, so there is time to step out of view.
- (void)requestPlateCaptureIn:(double)seconds;
// Seconds until capture starts, 0 while capturing, and -1 when idle.
- (double)plateCaptureRemaining;
// When capture starts, or 0 when idle, so a switch of model keeps a capture.
@property (atomic) double plateCaptureTime;
- (void)requestPlateForget;
// Whether frames in this frame's pixel format can be sent to the model.
- (BOOL)acceptsFrame:(const struct obs_source_frame *)frame;
- (BOOL)submitFrame:(const struct obs_source_frame *)frame;
// Copies the newest matte made from a frame no later than timestamp into dst,
// unless that matte is the one already copied (*generation).
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
	float *_changeGrid;
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

	// Owned by the model queue: the low-light lift.
	float _liftGain;
	float _liftTableGain;
	Pixel_8 _liftTable[256];
	Pixel_8 _liftAlpha[256];
	// Owned by the publish queue: the empty-background photo (BGRA at the
	// model size) and the blurred color difference to it.
	NSString *_platePath;
	uint8_t *_plate;
	uint8_t *_plateDiff;
	uint8_t *_plateDiffBlur;
	uint8_t *_plateNoise;
	uint8_t *_plateGap;
	uint8_t *_plateGapWide;
	uint8_t *_plateGapThin;
	uint32_t *_plateSum;
	uint32_t *_plateSumSquares;
	int _plateFrames;
	int _plateAdaptCount;
	float _plateGain[3];
	BOOL _plateValid;

	// Owned by the publish queue: the person's solid middle on the motion
	// grid, rebuilt from each raw matte by updateCoreFrom.
	uint8_t *_coreSource;
	uint8_t *_core;
	uint32_t _coreWidth;
	uint32_t _coreHeight;

	// Read and written on the OBS video thread, and on the publish queue, which clears it when the capture ends.
	_Atomic double _plateCaptureAt;
	_Atomic bool _plateForgetRequested;


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
	free(_changeGrid);
	free(_smoothed);
	free(_gridBGRA);
	free(_luma);
	free(_prevLuma);
	free(_diff);
	free(_motion);
	free(_plate);
	free(_plateDiff);
	free(_plateDiffBlur);
	free(_plateNoise);
	free(_plateGap);
	free(_plateGapWide);
	free(_plateGapThin);
	free(_plateSum);
	free(_plateSumSquares);
	free(_coreSource);
	free(_core);
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
		free(_changeGrid);
		_changeGrid = malloc(count * 2 * sizeof(float));
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

- (double)plateCaptureTime
{
	return _plateCaptureAt;
}

- (void)setPlateCaptureTime:(double)time
{
	_plateCaptureAt = time;
}

- (double)plateCaptureRemaining
{
	double at = _plateCaptureAt;
	return at > 0 ? MAX(0.0, at - CFAbsoluteTimeGetCurrent()) : -1.0;
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
	_plateNoise = malloc(count);
	memset(_plateNoise, kPlateNoiseDefault, count);
	_plateGap = malloc(count);
	_plateGapWide = malloc(count);
	_plateGapThin = malloc(count);
	_plateGain[0] = _plateGain[1] = _plateGain[2] = 1.0f;
}

// Adds a frame to the photo being captured. Once there are enough, stores
// their average as the photo and the spread of each pixel as its noise.
- (void)addPlateFrame:(const uint8_t *)frame stride:(size_t)stride width:(uint32_t)width height:(uint32_t)height
{
	size_t count = (size_t)width * height;
	if (!_plateSum) {
		_plateSum = calloc(count * 3, sizeof(uint32_t));
		_plateSumSquares = calloc(count, sizeof(uint32_t));
		_plateFrames = 0;
	}
	for (uint32_t y = 0; y < height; y++) {
		const uint8_t *p = frame + y * stride;
		uint32_t *sum = _plateSum + (size_t)y * width * 3;
		uint32_t *squares = _plateSumSquares + (size_t)y * width;
		for (uint32_t x = 0; x < width; x++, p += 4, sum += 3) {
			sum[0] += p[0];
			sum[1] += p[1];
			sum[2] += p[2];
			uint32_t total = (uint32_t)p[0] + p[1] + p[2];
			squares[x] += total * total;
		}
	}
	if (++_plateFrames < kPlateCaptureFrames)
		return;

	float n = (float)_plateFrames;
	for (size_t i = 0; i < count; i++) {
		const uint32_t *sum = _plateSum + i * 3;
		uint8_t *q = _plate + i * 4;
		q[0] = (uint8_t)((float)sum[0] / n + 0.5f);
		q[1] = (uint8_t)((float)sum[1] / n + 0.5f);
		q[2] = (uint8_t)((float)sum[2] / n + 0.5f);
		q[3] = 255;
		// Spread of the pixel's brightness, the mean of its three channels.
		float mean = (float)(sum[0] + sum[1] + sum[2]) / (3.0f * n);
		float variance = fmaxf(0.0f, (float)_plateSumSquares[i] / (9.0f * n) - mean * mean);
		_plateNoise[i] = (uint8_t)fminf(20.0f, sqrtf(variance) + 0.5f);
	}
	free(_plateSum);
	free(_plateSumSquares);
	_plateSum = NULL;
	_plateSumSquares = NULL;
	_plateCaptureAt = 0;
	_plateGain[0] = _plateGain[1] = _plateGain[2] = 1.0f;
	_plateValid = YES;
	[self savePlate];
	blog_rvm(LOG_INFO, "Captured background photo from %d frames", _plateFrames);
}

// Updates the gains that match the frame to the photo, from pixels the model
// is sure are background.
- (void)updatePlateGainFrom:(const uint8_t *)frame
		     stride:(size_t)stride
		      matte:(const uint8_t *)matte
		      width:(uint32_t)width
		     height:(uint32_t)height
{
	double live[3] = {0}, photo[3] = {0};
	size_t samples = 0;
	for (uint32_t y = 0; y < height; y += kPlateGainStep) {
		for (uint32_t x = 0; x < width; x += kPlateGainStep) {
			if (matte[(size_t)y * width + x] >= 8)
				continue;
			const uint8_t *p = frame + y * stride + (size_t)x * 4;
			const uint8_t *q = _plate + ((size_t)y * width + x) * 4;
			for (int c = 0; c < 3; c++) {
				live[c] += p[c];
				photo[c] += q[c];
			}
			samples++;
		}
	}
	if (samples < 200)
		return;
	for (int c = 0; c < 3; c++) {
		float target = live[c] > 0 ? (float)(photo[c] / live[c]) : 1.0f;
		target = fminf(kPlateGainMax, fmaxf(kPlateGainMin, target));
		_plateGain[c] += (target - _plateGain[c]) * kPlateGainRate;
	}
}

// Marks the person's solid middle on the motion grid: cells that are fully
// opaque in the model's matte, with small gaps filled, shrunk inward by
// kCoreShrink cells. The photo only cuts thin gaps inside it.
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
	// Filling grows the shape by kCoreGapFill - 1 cells across, so the shrink
	// takes that back as well.
	vImageMax_Planar8(&core_source, &core, NULL, 0, 0, kCoreGapFill, kCoreGapFill, kvImageEdgeExtend);
	vImagePixelCount shrink = kCoreShrink + kCoreGapFill - 1;
	vImageMin_Planar8(&core, &core_source, NULL, 0, 0, shrink, shrink, kvImageEdgeExtend);
	memcpy(_core, _coreSource, (size_t)gw * gh);
}

// Moves one color level of the photo one step toward the live picture.
static inline uint8_t adapt_level(uint8_t q, int d)
{
	if (d > 0 && q < 255)
		return (uint8_t)(q + 1);
	if (d < 0 && q > 0)
		return (uint8_t)(q - 1);
	return q;
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
		NSData *noise = [NSData dataWithContentsOfFile:[_platePath stringByAppendingString:@".noise"]];
		if (noise.length == (size_t)_modelWidth * _modelHeight)
			memcpy(_plateNoise, noise.bytes, noise.length);
		blog_rvm(LOG_INFO, "Loaded background photo");
	}
	CGImageRelease(image);
}

- (void)savePlate
{
	if (!_platePath.length)
		return;
	NSData *noise = [NSData dataWithBytesNoCopy:_plateNoise
					     length:(size_t)_modelWidth * _modelHeight
				       freeWhenDone:NO];
	[noise writeToFile:[_platePath stringByAppendingString:@".noise"] atomically:YES];
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

// Cuts out pixels that match the empty-background photo, apart from the solid
// middle of the person, and lets the photo follow slow lighting changes.
// Runs on the publish queue with the model's raw matte, before smoothing.
- (void)applyPlateTo:(uint8_t *)matte width:(uint32_t)width height:(uint32_t)height frame:(CVPixelBufferRef)bgra
{
	if (_plateForgetRequested) {
		_plateForgetRequested = false;
		_plateValid = NO;
		free(_plateSum);
		free(_plateSumSquares);
		_plateSum = NULL;
		_plateSumSquares = NULL;
		if (_platePath.length) {
			NSFileManager *files = [NSFileManager defaultManager];
			[files removeItemAtPath:_platePath error:nil];
			[files removeItemAtPath:[_platePath stringByAppendingString:@".noise"] error:nil];
		}
		blog_rvm(LOG_INFO, "Forgot background photo");
	}
	if (width != _modelWidth || height != _modelHeight || CVPixelBufferGetWidth(bgra) != width ||
	    CVPixelBufferGetHeight(bgra) != height)
		return;

	CVPixelBufferLockBaseAddress(bgra, kCVPixelBufferLock_ReadOnly);
	const uint8_t *frame = CVPixelBufferGetBaseAddress(bgra);
	size_t stride = CVPixelBufferGetBytesPerRow(bgra);

	double capture_at = _plateCaptureAt;
	BOOL capturing = capture_at > 0 && CFAbsoluteTimeGetCurrent() >= capture_at;
	if (!capturing && _plateSum) {
		free(_plateSum);
		free(_plateSumSquares);
		_plateSum = NULL;
		_plateSumSquares = NULL;
	}
	if (capturing) {
		[self allocatePlateBuffers];
		[self addPlateFrame:frame stride:stride width:width height:height];
	}
	if (!_plateValid || !self.usePlate) {
		CVPixelBufferUnlockBaseAddress(bgra, kCVPixelBufferLock_ReadOnly);
		return;
	}

	uint32_t gw = width / kMotionGridScale;
	uint32_t gh = height / kMotionGridScale;
	BOOL have_motion = _motionValid && _gridWidth == gw && _gridHeight == gh;

	[self updatePlateGainFrom:frame stride:stride matte:matte width:width height:height];
	const float gain_b = _plateGain[0], gain_g = _plateGain[1], gain_r = _plateGain[2];
	const BOOL adapt = ++_plateAdaptCount % kPlateAdaptEvery == 0;

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
			int db = MIN(255, (int)((float)p[0] * gain_b + 0.5f)) - q[0];
			int dg = MIN(255, (int)((float)p[1] * gain_g + 0.5f)) - q[1];
			int dr = MIN(255, (int)((float)p[2] * gain_r + 0.5f)) - q[2];
			int largest = MAX(abs(db), MAX(abs(dg), abs(dr)));
			d[x] = (uint8_t)MIN(255, largest);
			if (adapt && motion_row && m[x] < 8 && largest < kPlateAdaptLimit &&
			    motion_row[MIN(x / kMotionGridScale, gw - 1)] <= (uint8_t)kMotionNoiseFloor) {
				q[0] = adapt_level(q[0], db);
				q[1] = adapt_level(q[1], dg);
				q[2] = adapt_level(q[2], dr);
			}
		}
	}
	CVPixelBufferUnlockBaseAddress(bgra, kCVPixelBufferLock_ReadOnly);

	vImage_Buffer diff = {.data = _plateDiff, .height = height, .width = width, .rowBytes = width};
	vImage_Buffer blur = {.data = _plateDiffBlur, .height = height, .width = width, .rowBytes = width};
	vImageBoxConvolve_Planar8(&diff, &blur, NULL, 0, 0, kPlateBlur, kPlateBlur, 0, kvImageEdgeExtend);


	// Find thin gaps: exact matches, minus wide areas and single specks.
	size_t count = (size_t)width * height;
	for (size_t i = 0; i < count; i++) {
		const uint8_t *q = _plate + i * 4;
		float limit = kPlateGapMatch + kPlateNoiseScale * (float)_plateNoise[i];
		BOOL bright = MAX(q[0], MAX(q[1], q[2])) >= kPlateGapMinLevel;
		_plateGap[i] = bright && matte[i] && (float)_plateDiffBlur[i] <= limit ? 255 : 0;
	}
	vImage_Buffer gap = {.data = _plateGap, .height = height, .width = width, .rowBytes = width};
	vImage_Buffer wide = {.data = _plateGapWide, .height = height, .width = width, .rowBytes = width};
	vImage_Buffer thin = {.data = _plateGapThin, .height = height, .width = width, .rowBytes = width};
	// Wide areas are found on the motion grid, which is much cheaper than a
	// kPlateGapWidth window at full size.
	vImagePixelCount grid_window = (kPlateGapWidth / kMotionGridScale) | 1;
	for (uint32_t y = 0; y < gh; y++) {
		const uint8_t *row = _plateGap + ((size_t)y * kMotionGridScale + kMotionGridScale / 2) * width;
		for (uint32_t x = 0; x < gw; x++)
			_plateGapThin[(size_t)y * gw + x] = row[x * kMotionGridScale + kMotionGridScale / 2];
	}
	vImage_Buffer grid = {.data = _plateGapThin, .height = gh, .width = gw, .rowBytes = gw};
	vImage_Buffer eroded = {.data = _plateGapWide, .height = gh, .width = gw, .rowBytes = gw};
	// Close small misses first, so a speckled wide match still counts as wide.
	vImageMax_Planar8(&grid, &eroded, NULL, 0, 0, 3, 3, kvImageEdgeExtend);
	vImageMin_Planar8(&eroded, &grid, NULL, 0, 0, 3, 3, kvImageEdgeExtend);
	vImageMin_Planar8(&grid, &eroded, NULL, 0, 0, grid_window, grid_window, kvImageEdgeExtend);
	vImageMax_Planar8(&eroded, &grid, NULL, 0, 0, grid_window, grid_window, kvImageEdgeExtend);
	vImageMax_Planar8(&grid, &eroded, NULL, 0, 0, 3, 3, kvImageEdgeExtend); // and the rims of wide areas

	// Keep gaps to the edge of the upper half of the person.
	size_t grid_count = (size_t)gw * gh;
	uint8_t *person = _plateGapWide + grid_count;
	uint8_t *middle = _plateGapWide + grid_count * 2;
	uint32_t first_row = gh, last_row = 0;
	for (uint32_t y = 0; y < gh; y++) {
		const uint8_t *row = matte + ((size_t)y * kMotionGridScale + kMotionGridScale / 2) * width;
		for (uint32_t x = 0; x < gw; x++) {
			BOOL inside = row[x * kMotionGridScale + kMotionGridScale / 2] >= 128;
			person[(size_t)y * gw + x] = inside ? 255 : 0;
			if (inside) {
				first_row = MIN(first_row, y);
				last_row = MAX(last_row, y);
			}
		}
	}
	uint32_t row_limit = first_row + (last_row > first_row ? (last_row - first_row) / 2 : 0);
	vImage_Buffer person_grid = {.data = person, .height = gh, .width = gw, .rowBytes = gw};
	vImage_Buffer middle_grid = {.data = middle, .height = gh, .width = gw, .rowBytes = gw};
	vImageMin_Planar8(&person_grid, &middle_grid, NULL, 0, 0, kPlateGapReach, kPlateGapReach, kvImageEdgeExtend);

	for (uint32_t y = 0; y < height; y++) {
		uint8_t *g = _plateGap + (size_t)y * width;
		uint32_t cell_y = MIN(y / kMotionGridScale, gh - 1);
		if (cell_y > row_limit) {
			memset(g, 0, width);
			continue;
		}
		const uint8_t *wide_row = _plateGapWide + (size_t)cell_y * gw;
		const uint8_t *middle_row = middle + (size_t)cell_y * gw;
		for (uint32_t x = 0; x < width; x++) {
			uint32_t cell_x = MIN(x / kMotionGridScale, gw - 1);
			if (wide_row[cell_x] || middle_row[cell_x])
				g[x] = 0;
		}
	}
	vImageMin_Planar8(&gap, &wide, NULL, 0, 0, kPlateGapSpeck, kPlateGapSpeck, kvImageEdgeExtend);
	vImageMax_Planar8(&wide, &thin, NULL, 0, 0, kPlateGapSpeck, kPlateGapSpeck, kvImageEdgeExtend);

	const float inv_range = 1.0f / (kPlateMatchHigh - kPlateMatchLow);
	for (uint32_t y = 0; y < height; y++) {
		uint8_t *m = matte + (size_t)y * width;
		const uint8_t *b = _plateDiffBlur + (size_t)y * width;
		const uint8_t *noise = _plateNoise + (size_t)y * width;
		const uint8_t *gap_row = _plateGapThin + (size_t)y * width;
		const uint8_t *core_row = _core + (size_t)MIN(y / kMotionGridScale, gh - 1) * gw;
		for (uint32_t x = 0; x < width; x++) {
			if (gap_row[x]) {
				m[x] = 0;
				continue;
			}
			float low = kPlateMatchLow + kPlateNoiseScale * (float)noise[x];
			float value = (float)b[x] - low;
			if (!m[x] || m[x] >= kPlateSureMatte || value * inv_range >= 1.0f ||
			    core_row[MIN(x / kMotionGridScale, gw - 1)])
				continue;
			float t = fmaxf(0.0f, value * inv_range);
			float keep = t * t * (3.0f - 2.0f * t);
			m[x] = (uint8_t)((float)m[x] * keep + 0.5f);
		}
	}
}

// Returns a brightened copy of a dark frame for the model, or NULL when the
// frame is bright enough to use as it is.
- (CVPixelBufferRef)newLiftedFrom:(CVPixelBufferRef)bgra CF_RETURNS_RETAINED
{
	static float linear[256];
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		for (int i = 0; i < 256; i++)
			linear[i] = powf((float)i / 255.0f, 2.2f);
	});

	CVPixelBufferLockBaseAddress(bgra, kCVPixelBufferLock_ReadOnly);
	const uint8_t *base = CVPixelBufferGetBaseAddress(bgra);
	size_t width = CVPixelBufferGetWidth(bgra), height = CVPixelBufferGetHeight(bgra);
	size_t row_bytes = CVPixelBufferGetBytesPerRow(bgra);
	double sum = 0.0;
	size_t samples = 0;
	for (size_t y = 0; y < height; y += kLiftSampleStep) {
		const uint8_t *row = base + y * row_bytes;
		for (size_t x = 0; x < width; x += kLiftSampleStep) {
			const uint8_t *p = row + x * 4;
			sum += linear[p[0]] + linear[p[1]] + linear[p[2]];
			samples += 3;
		}
	}
	CVPixelBufferUnlockBaseAddress(bgra, kCVPixelBufferLock_ReadOnly);

	float mean = samples ? (float)(sum / (double)samples) : kLiftTarget;
	float target = fminf(kLiftMax, fmaxf(1.0f, kLiftTarget / fmaxf(mean, 0.0001f)));
	if (_liftGain < 1.0f)
		_liftGain = target;
	_liftGain += (target - _liftGain) * kLiftRate;
	if (_liftGain < kLiftMinimum)
		return NULL;

	if (fabsf(_liftGain - _liftTableGain) > 0.01f) {
		for (int i = 0; i < 256; i++) {
			float lifted = fminf(1.0f, linear[i] * _liftGain);
			_liftTable[i] = (Pixel_8)(powf(lifted, 1.0f / 2.2f) * 255.0f + 0.5f);
			_liftAlpha[i] = (Pixel_8)i;
		}
		_liftTableGain = _liftGain;
	}

	CVPixelBufferRef out = NULL;
	if (CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, _bgraPool, &out) != kCVReturnSuccess)
		return NULL;
	CVPixelBufferLockBaseAddress(bgra, kCVPixelBufferLock_ReadOnly);
	CVPixelBufferLockBaseAddress(out, 0);
	vImage_Buffer src = {.data = CVPixelBufferGetBaseAddress(bgra), .height = height, .width = width,
			     .rowBytes = row_bytes};
	vImage_Buffer dst = {.data = CVPixelBufferGetBaseAddress(out), .height = height, .width = width,
			     .rowBytes = CVPixelBufferGetBytesPerRow(out)};
	// The buffer is BGRA, and the table's channels run in memory order.
	vImageTableLookUp_ARGB8888(&src, &dst, _liftTable, _liftTable, _liftTable, _liftAlpha, kvImageNoFlags);
	CVPixelBufferUnlockBaseAddress(out, 0);
	CVPixelBufferUnlockBaseAddress(bgra, kCVPixelBufferLock_ReadOnly);
	return out;
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

	CVPixelBufferRef lifted = [self newLiftedFrom:bgra];
	NSMutableDictionary *features = [_states mutableCopy];
	features[@"src"] = [MLFeatureValue featureValueWithPixelBuffer:lifted ? lifted : bgra];
	NSError *error = nil;
	MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:features
												   error:&error];
	id<MLFeatureProvider> output = provider ? [_model predictionFromFeatures:provider error:&error] : nil;
	if (lifted)
		CVPixelBufferRelease(lifted);
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

// Damps small changes in the matte where the camera picture is still. Each
// pixel moves toward the new value by a weight that is small where nothing moved
// (so the model's wavering settles) and 1 where the picture moved or the change
// is large over a block (so the cutout follows at once). change_grid holds two
// grids of floats, used as scratch space. raw and previous are the unblurred mattes of this
// frame and the last, used only for the edge flicker statistic. Returns the
// summed absolute change, for the overall flicker statistic.
static double smooth_matte(float *smoothed, const uint8_t *incoming, const uint8_t *raw, const uint8_t *previous,
			   uint32_t width, uint32_t height, const uint8_t *motion, uint32_t grid_width,
			   uint32_t grid_height, float stability, float *change_grid, struct edge_flicker *edge)
{
	const float still_weight = 1.0f - 0.95f * stability;
	const float inv_threshold = 1.0f / kMotionThreshold;
	const float inv_follow = 1.0f / (kSteadyFollowHigh - kSteadyFollowLow);
	double change = 0.0;

	// This sums the change in each grid cell, and then averages each block of
	// three by three cells. Blocks at the picture's edge have fewer cells.
	size_t grid_count = (size_t)grid_width * grid_height;
	float *cells = change_grid;
	float *blocks = change_grid + grid_count;
	memset(cells, 0, grid_count * sizeof(float));
	for (uint32_t y = 0; y < height; y++) {
		float *cell_row = cells + (size_t)MIN(y / kMotionGridScale, grid_height - 1) * grid_width;
		const float *s = smoothed + (size_t)y * width;
		const uint8_t *in = incoming + (size_t)y * width;
		for (uint32_t x = 0; x < width; x++)
			cell_row[MIN(x / kMotionGridScale, grid_width - 1)] += fabsf((float)in[x] - s[x]);
	}
	const float inv_cell = 1.0f / (float)(kMotionGridScale * kMotionGridScale);
	for (uint32_t y = 0; y < grid_height; y++) {
		for (uint32_t x = 0; x < grid_width; x++) {
			float sum = 0.0f;
			float visited = 0.0f;
			for (uint32_t cy = y ? y - 1 : 0; cy <= MIN(y + 1, grid_height - 1); cy++) {
				for (uint32_t cx = x ? x - 1 : 0; cx <= MIN(x + 1, grid_width - 1); cx++) {
					sum += cells[(size_t)cy * grid_width + cx];
					visited += 1.0f;
				}
			}
			blocks[(size_t)y * grid_width + x] = sum * inv_cell / visited;
		}
	}

	for (uint32_t y = 0; y < height; y++) {
		size_t grid_row = (size_t)MIN(y / kMotionGridScale, grid_height - 1) * grid_width;
		const uint8_t *motion_row = motion + grid_row;
		const float *block_row = blocks + grid_row;
		float *s = smoothed + (size_t)y * width;
		const uint8_t *in = incoming + (size_t)y * width;
		const uint8_t *r = raw + (size_t)y * width;
		const uint8_t *p = previous + (size_t)y * width;
		for (uint32_t x = 0; x < width; x++) {
			uint32_t cell = MIN(x / kMotionGridScale, grid_width - 1);
			float m = (float)motion_row[cell];
			float moving = fminf(1.0f, fmaxf(0.0f, (m - kMotionNoiseFloor) * inv_threshold));
			float d = (float)in[x] - s[x];
			float large = fminf(1.0f, fmaxf(0.0f, (block_row[cell] - kSteadyFollowLow) * inv_follow));
			float w = still_weight + (1.0f - still_weight) * fmaxf(moving, large);
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
					     _gridWidth, _gridHeight, self.stability, _changeGrid, &_statsEdge) /
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
			_slots[s] = malloc(count);
			_slotTimestamps[s] = 0;
			_slotGenerations[s] = 0;
		}
		_matteWidth = width;
		_matteHeight = height;
	}
	_matteGeneration++;
	int slot = (int)(_matteGeneration % MATTE_SLOTS);
	uint8_t *dst = _slots[slot];
	for (size_t i = 0; i < count; i++)
		dst[i] = (uint8_t)(_smoothed[i] + 0.5f);
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

	// White balance: move the plain background toward neutral, keeping green fixed.
	float balance[3] = {1.0f, 1.0f, 1.0f};
	if (neutral_n >= (uint64_t)(kAutoBalanceMinShare * (float)total) && neutral[0] > 0 && neutral[2] > 0) {
		balance[0] = fminf(kAutoBalanceMax, fmaxf(kAutoBalanceMin, (float)(neutral[1] / neutral[0])));
		balance[2] = fminf(kAutoBalanceMax, fmaxf(kAutoBalanceMin, (float)(neutral[1] / neutral[2])));
		balance[0] = 1.0f + kAutoBalanceStrength * (balance[0] - 1.0f);
		balance[2] = 1.0f + kAutoBalanceStrength * (balance[2] - 1.0f);
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
		size_t size = (size_t)_matteWidth * _matteHeight;
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
// without capturing the camera image itself.
- (void)writeMatteSnapshotTo:(NSString *)path
{
	// Copy the matte under the lock and encode outside it, so the video
	// thread never waits for PNG encoding.
	[_matteLock lock];
	uint32_t width = _matteWidth, height = _matteHeight;
	size_t count = (size_t)width * height;
	uint8_t *matte = malloc(count);
	memcpy(matte, _slots[_matteGeneration % MATTE_SLOTS], count);
	[_matteLock unlock];
	write_gray_png(matte, width, height, path);
	free(matte);
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
	int countdown_idle;
	int countdown_capturing;
	long countdown_frames;
	volatile long capture_frames;
	volatile bool plate_cancel_requested;
	char *plate_path;
	volatile long capture_ms;
	int countdown_label;
	obs_source_t *context;
	gs_effect_t *effect;
	gs_eparam_t *param_matte;
	gs_eparam_t *param_low;
	gs_eparam_t *param_high;
	gs_eparam_t *param_gain;
	gs_eparam_t *param_lift;
	gs_eparam_t *param_edge_reach;
	gs_eparam_t *param_room_wrap;
	gs_eparam_t *param_room_light;
	gs_eparam_t *param_room_mean;
	gs_eparam_t *param_room_blend;
	gs_eparam_t *param_room_reach;

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
	bool clean_edges;

	// "Blend into the room". All of this is used on the video thread.
	char *room_name;
	obs_weak_source_t *room;
	float room_blend;
	gs_texrender_t *room_maps[ROOM_MAP_COUNT];
	uint64_t room_next_lookup;
	uint64_t room_frame_time;
	bool room_ready;

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
		ready.plateCaptureTime = engine_of(f).plateCaptureTime;
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
	// Each filter has its own photo, so two cameras cannot overwrite each other.
	char *old_plate_path = obs_module_config_path("background-photo.png");

	NSString *model_path = @(path);
	NSString *plate = f->plate_path ? @(f->plate_path) : nil;
	NSFileManager *files = [NSFileManager defaultManager];
	if (plate && old_plate_path && ![files fileExistsAtPath:plate])
		[files moveItemAtPath:@(old_plate_path) toPath:plate error:nil];
	bfree(path);
	bfree(old_plate_path);

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
	f->clean_edges = obs_data_get_bool(settings, "clean_edges");
	f->room_blend = (float)obs_data_get_double(settings, "room_blend");
	const char *room_name = obs_data_get_string(settings, "room_source");
	if (strcmp(room_name, f->room_name ? f->room_name : "") != 0) {
		bfree(f->room_name);
		f->room_name = bstrdup(room_name);
		obs_weak_source_release(f->room);
		f->room = NULL;
		f->room_next_lookup = 0;
		f->room_ready = false;
	}
	f->sync = obs_data_get_bool(settings, "sync_video");
	f->use_plate = obs_data_get_bool(settings, "use_plate");
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
	// a saved background photo.
	const char *snapshot = obs_data_get_string(settings, "debug_snapshot");
	if (snapshot && *snapshot) {
		NSString *name = @(snapshot);
		if ([name isEqualToString:name.lastPathComponent] && ![name hasPrefix:@"."] &&
		    [name.pathExtension.lowercaseString isEqualToString:@"png"] &&
		    ![name.lowercaseString hasPrefix:@"background-photo"]) {
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
	f->capture_ms = -1;
	NSString *plate_name = [NSString stringWithFormat:@"background-photo-%s.png", obs_source_get_uuid(context)];
	f->plate_path = obs_module_config_path(plate_name.UTF8String);
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
		f->param_edge_reach = gs_effect_get_param_by_name(f->effect, "edge_reach");
		f->param_room_wrap = gs_effect_get_param_by_name(f->effect, "room_wrap");
		f->param_room_light = gs_effect_get_param_by_name(f->effect, "room_light");
		f->param_room_mean = gs_effect_get_param_by_name(f->effect, "room_mean");
		f->param_room_blend = gs_effect_get_param_by_name(f->effect, "room_blend");
		f->param_room_reach = gs_effect_get_param_by_name(f->effect, "room_reach");
	}
	obs_leave_graphics();
	bfree(effect_path);

	if (!f->effect) {
		blog_rvm(LOG_ERROR, "Could not load rvm-matting.effect");
		CFBridgingRelease(f->loader);
		bfree(f->plate_path);
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
	for (size_t i = 0; i < ROOM_MAP_COUNT; i++)
		gs_texrender_destroy(f->room_maps[i]);
	obs_leave_graphics();

	obs_weak_source_release(f->room);
	bfree(f->room_name);
	bfree(f->matte_copy);
	bfree(f->model_name);
	bfree(f->snapshot_name);
	bfree(f->plate_path);
	bfree(f);
}

// Frames still held when the filter is removed go back to the camera source,
// as in OBS's own async delay filter.
static void rvm_filter_remove(void *data, obs_source_t *parent)
{
	// OBS calls this even when rvm_create failed and left no data.
	if (data)
		release_held_frames(data, parent);
}

static struct obs_source_frame *rvm_filter_video(void *data, struct obs_source_frame *frame)
{
	struct rvm_filter *f = data;
	obs_source_t *parent = obs_filter_get_parent(f->context);
	adopt_ready_engine(f);
	bool accepted = false;
	if (f->engine) {
		// Stop any old capture first, so a new request always survives.
		if (os_atomic_set_bool(&f->plate_forget_requested, false)) {
			engine_of(f).plateCaptureTime = 0;
			[engine_of(f) requestPlateForget];
		}
		if (os_atomic_set_bool(&f->plate_cancel_requested, false))
			engine_of(f).plateCaptureTime = 0;
		if (os_atomic_set_bool(&f->plate_capture_requested, false))
			[engine_of(f) requestPlateCaptureIn:kPlateCaptureDelaySeconds];
		os_atomic_set_long(&f->capture_frames, os_atomic_load_long(&f->capture_frames) + 1);
		double remaining = [engine_of(f) plateCaptureRemaining];
		os_atomic_set_long(&f->capture_ms, remaining < 0.0 ? -1 : (long)(remaining * 1000.0));
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

struct room_tree {
	obs_source_t *camera;
	bool found;
};

static void rvm_find_camera(obs_source_t *parent, obs_source_t *child, void *data)
{
	UNUSED_PARAMETER(parent);
	struct room_tree *tree = data;
	if (child == tree->camera)
		tree->found = true;
}

// A room that shows the camera would draw the person into their own light, and
// nesting a scene item's render inside itself upsets the render target.
static bool room_contains(obs_source_t *room, obs_source_t *camera)
{
	struct room_tree tree = {camera, room == camera};
	if (camera && !tree.found)
		obs_source_enum_full_tree(room, rvm_find_camera, &tree);
	return tree.found;
}

// Returns the chosen room background with a new reference, or NULL.
static obs_source_t *room_source(struct rvm_filter *f)
{
	obs_source_t *room = f->room ? obs_weak_source_get_source(f->room) : NULL;
	if (room || !f->room_name || !*f->room_name)
		return room;

	uint64_t now = os_gettime_ns();
	if (now < f->room_next_lookup)
		return NULL;
	f->room_next_lookup = now + kRoomLookupNs;
	room = obs_get_source_by_name(f->room_name);
	if (room && room_contains(room, obs_filter_get_parent(f->context))) {
		obs_source_release(room);
		room = NULL;
	}
	obs_weak_source_release(f->room);
	f->room = room ? obs_source_get_weak_source(room) : NULL;
	return room;
}

// Renders the room background into the blurred maps, once per video frame.
// This must run outside obs_source_process_filter_begin and _end.
static void render_room_maps(struct rvm_filter *f)
{
	uint64_t frame_time = obs_get_video_frame_time();
	if (frame_time == f->room_frame_time)
		return;
	f->room_frame_time = frame_time;
	f->room_ready = false;

	obs_source_t *room = room_source(f);
	if (!room)
		return;
	uint32_t room_width = obs_source_get_width(room);
	uint32_t room_height = obs_source_get_height(room);
	if (room == obs_filter_get_parent(f->context) || !room_width || !room_height) {
		obs_source_release(room);
		return;
	}

	gs_effect_t *copy = obs_get_base_effect(OBS_EFFECT_DEFAULT);
	gs_eparam_t *copy_image = gs_effect_get_param_by_name(copy, "image");
	bool drawn = true;
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	for (size_t i = 0; i < ROOM_MAP_COUNT && drawn; i++) {
		uint32_t width = kRoomMaps[i].width, height = kRoomMaps[i].height;
		if (!f->room_maps[i])
			f->room_maps[i] = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
		gs_texrender_reset(f->room_maps[i]);
		drawn = gs_texrender_begin(f->room_maps[i], width, height);
		if (!drawn)
			break;
		struct vec4 clear;
		vec4_zero(&clear);
		gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
		if (i == 0) {
			gs_ortho(0.0f, (float)room_width, 0.0f, (float)room_height, -100.0f, 100.0f);
			obs_source_video_render(room);
		} else {
			gs_ortho(0.0f, (float)width, 0.0f, (float)height, -100.0f, 100.0f);
			gs_texture_t *larger = gs_texrender_get_texture(f->room_maps[i - 1]);
			gs_effect_set_texture(copy_image, larger);
			while (gs_effect_loop(copy, "Draw"))
				gs_draw_sprite(larger, 0, width, height);
		}
		gs_texrender_end(f->room_maps[i]);
	}
	gs_blend_state_pop();
	obs_source_release(room);
	f->room_ready = drawn;
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
				f->matte_tex = gs_texture_create(f->matte_width, f->matte_height, GS_R8, 1, NULL,
								 GS_DYNAMIC);
			}
			if (f->matte_tex)
				gs_texture_set_image(f->matte_tex, f->matte_copy, f->matte_width, false);
		}
	}

	if (!f->matte_tex || !f->engine) {
		obs_source_skip_video_filter(f->context);
		return;
	}

	// Premultiplied blending and color-space choice as in OBS's own chroma key
	// filter. Filters that change the picture's size must not come before this
	// one, because the matte is made from the whole camera frame.
	const enum gs_color_space preferred_spaces[] = {
		GS_CS_SRGB,
		GS_CS_SRGB_16F,
		GS_CS_709_EXTENDED,
	};
	const enum gs_color_space space = obs_source_get_color_space(obs_filter_get_target(f->context),
								     OBS_COUNTOF(preferred_spaces), preferred_spaces);
	// Like the color math, the room light assumes sRGB-encoded 8-bit input.
	if (f->room_blend > 0.0f && space == GS_CS_SRGB)
		render_room_maps(f);
	const bool use_room = f->room_blend > 0.0f && space == GS_CS_SRGB && f->room_ready;

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
	// The reach is the same distance across and down on a picture of any shape.
	struct vec2 edge_reach;
	vec2_zero(&edge_reach);
	if (f->clean_edges)
		vec2_set(&edge_reach, kEdgeReach * (float)f->matte_height / (float)f->matte_width, kEdgeReach);
	gs_effect_set_vec2(f->param_edge_reach, &edge_reach);
	// The shader skips the room maps at a blend of 0, but they must still be set.
	gs_effect_set_texture(f->param_room_wrap,
			      use_room ? gs_texrender_get_texture(f->room_maps[kRoomWrapMap]) : f->matte_tex);
	gs_effect_set_texture(f->param_room_light,
			      use_room ? gs_texrender_get_texture(f->room_maps[kRoomLightMap]) : f->matte_tex);
	gs_effect_set_texture(f->param_room_mean,
			      use_room ? gs_texrender_get_texture(f->room_maps[kRoomMeanMap]) : f->matte_tex);
	gs_effect_set_float(f->param_room_blend, use_room ? f->room_blend : 0.0f);
	struct vec2 room_reach;
	vec2_set(&room_reach, kRoomEdgeReach * (float)f->matte_height / (float)f->matte_width, kRoomEdgeReach);
	gs_effect_set_vec2(f->param_room_reach, &room_reach);

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
	obs_data_set_default_bool(settings, "clean_edges", true);
	obs_data_set_default_string(settings, "room_source", "");
	obs_data_set_default_double(settings, "room_blend", 0.5);
	obs_data_set_default_bool(settings, "sync_video", true);
	obs_data_set_default_bool(settings, "use_plate", false);
	obs_data_set_default_bool(settings, "auto_color", false);
}

// Plays one countdown tick. A sound that is still playing refuses to start
// again, so each tick is its own copy, kept alive in a small ring.
static void rvm_play_tick(void)
{
	static NSSound *ticks[8];
	static int next;
	NSSound *tick = [[NSSound soundNamed:@"Tink"] copy];
	ticks[next] = tick;
	next = (next + 1) % 8;
	[tick play];
}

// Counts down on the capture button and with sounds, because whoever clicked
// has to walk out of view and may not see the screen. countdown_label is 0
// when idle, -1 after a failed capture, 1 while capturing, and otherwise the
// seconds left plus 1. Runs on
// the main queue and stops by itself when the capture has finished.
static void rvm_countdown_tick(obs_weak_source_t *weak, int ticks, bool started)
{
	dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 100 * NSEC_PER_MSEC), dispatch_get_main_queue(), ^{
		obs_source_t *source = obs_weak_source_get_source(weak);
		if (!source) {
			obs_weak_source_release(weak);
			return;
		}
		struct rvm_filter *f = obs_obj_get_data(source);
		// Forget background photo cancels the countdown.
		if (f->countdown_label <= 0) {
			obs_source_release(source);
			obs_weak_source_release(weak);
			return;
		}
		// The engine belongs to the video thread, so read what it published.
		long capture_ms = os_atomic_load_long(&f->capture_ms);
		double remaining = capture_ms < 0 ? -1.0 : (double)capture_ms / 1000.0;
		bool now_started = started || remaining >= 0.0;
		bool finished = started && remaining < 0.0;

		// Give up if frames stop, the capture never starts or never ends,
		// and cancel it so it cannot fire later with someone in view.
		long frames = os_atomic_load_long(&f->capture_frames);
		f->countdown_idle = frames == f->countdown_frames ? f->countdown_idle + 1 : 0;
		f->countdown_frames = frames;
		f->countdown_capturing += capture_ms == 0;
		bool abandoned = (!now_started && ticks > 50) || f->countdown_idle > 30 || f->countdown_capturing > 100;
		if (finished || abandoned) {
			f->countdown_label = abandoned ? -1 : 0;
			if (abandoned)
				blog_rvm(LOG_WARNING, "Background photo capture did not finish");
			os_atomic_set_bool(&f->plate_capture_requested, false);
			if (abandoned)
				os_atomic_set_bool(&f->plate_cancel_requested, true);
			obs_source_update_properties(source);
			[[NSSound soundNamed:finished ? @"Glass" : @"Basso"] play];
			obs_source_release(source);
			obs_weak_source_release(weak);
			return;
		}
		int label = remaining < 0.0 ? f->countdown_label : (int)ceil(remaining) + 1;
		if (label != f->countdown_label) {
			f->countdown_label = label;
			if (label > 1)
				rvm_play_tick();
			obs_source_update_properties(source);
		}
		obs_source_release(source);
		rvm_countdown_tick(weak, ticks + 1, now_started);
	});
}

static bool rvm_capture_plate_clicked(obs_properties_t *props, obs_property_t *property, void *data)
{
	UNUSED_PARAMETER(props);
	UNUSED_PARAMETER(property);
	struct rvm_filter *f = data;
	if (!f || f->countdown_label > 0)
		return false; // not created, or already counting down or capturing

	// Taking a photo means wanting to use it.
	obs_data_t *settings = obs_source_get_settings(f->context);
	if (!obs_data_get_bool(settings, "use_plate")) {
		obs_data_set_bool(settings, "use_plate", true);
		obs_source_update(f->context, settings);
	}
	obs_data_release(settings);

	os_atomic_set_bool(&f->plate_cancel_requested, false);
	os_atomic_set_long(&f->capture_ms, -1);
	os_atomic_set_bool(&f->plate_capture_requested, true);
	f->countdown_label = (int)kPlateCaptureDelaySeconds + 1;
	f->countdown_idle = 0;
	f->countdown_capturing = 0;
	rvm_play_tick();
	obs_source_update_properties(f->context);
	rvm_countdown_tick(obs_source_get_weak_source(f->context), 0, false);
	return false;
}

static bool rvm_forget_plate_clicked(obs_properties_t *props, obs_property_t *property, void *data)
{
	UNUSED_PARAMETER(props);
	UNUSED_PARAMETER(property);
	struct rvm_filter *f = data;
	if (!f)
		return false;
	os_atomic_set_bool(&f->plate_forget_requested, true);
	os_atomic_set_bool(&f->plate_capture_requested, false);
	f->countdown_label = 0;
	// Delete it now too, because the engine only acts on the next frame.
	if (f->plate_path) {
		NSString *path = @(f->plate_path);
		NSFileManager *files = [NSFileManager defaultManager];
		[files removeItemAtPath:path error:nil];
		[files removeItemAtPath:[path stringByAppendingString:@".noise"] error:nil];
	}
	obs_source_update_properties(f->context);
	return false;
}

// Sets a property's "?" tooltip. OBS shows plain text on one long line, but
// word-wraps rich text.
static void rvm_set_help(obs_property_t *property, const char *key)
{
	char text[1024];
	snprintf(text, sizeof(text), "<qt>%s</qt>", obs_module_text(key));
	obs_property_set_long_description(property, text);
}

struct room_list {
	obs_property_t *property;
	obs_source_t *parent;
};

// Lists every input with a picture that does not show the camera this filter is
// on. OBS does not offer scenes here, and groups are left out.
static bool rvm_add_room_source(void *data, obs_source_t *source)
{
	struct room_list *list = data;
	const char *name = obs_source_get_name(source);
	if (name && !obs_source_is_group(source) && (obs_source_get_output_flags(source) & OBS_SOURCE_VIDEO) &&
	    !room_contains(source, list->parent))
		obs_property_list_add_string(list->property, name, name);
	return true;
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

	rvm_set_help(model, "Model.Description");
	rvm_set_help(units, "ComputeUnits.Description");
	obs_property_t *low = obs_properties_add_float_slider(props, "alpha_low", obs_module_text("AlphaLow"), 0.0, 0.5,
							      0.01);
	rvm_set_help(low, "AlphaLow.Description");
	obs_property_t *high = obs_properties_add_float_slider(props, "alpha_high", obs_module_text("AlphaHigh"), 0.5,
							       1.0, 0.01);
	rvm_set_help(high, "AlphaHigh.Description");
	obs_property_t *stability = obs_properties_add_float_slider(props, "edge_stability",
								     obs_module_text("EdgeStability"), 0.0, 1.0, 0.05);
	rvm_set_help(stability, "EdgeStability.Description");
	obs_property_t *clean_edges = obs_properties_add_bool(props, "clean_edges", obs_module_text("CleanEdges"));
	rvm_set_help(clean_edges, "CleanEdges.Description");
	obs_property_t *sync = obs_properties_add_bool(props, "sync_video", obs_module_text("SyncVideo"));
	rvm_set_help(sync, "SyncVideo.Description");

	obs_property_t *auto_color = obs_properties_add_bool(props, "auto_color", obs_module_text("AutoColor"));
	rvm_set_help(auto_color, "AutoColor.Description");

	obs_property_t *room = obs_properties_add_list(props, "room_source", obs_module_text("RoomSource"),
						       OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(room, obs_module_text("RoomSource.None"), "");
	const struct rvm_filter *room_filter = data;
	struct room_list room_list = {room, room_filter ? obs_filter_get_parent(room_filter->context) : NULL};
	obs_enum_sources(rvm_add_room_source, &room_list);
	rvm_set_help(room, "RoomSource.Description");
	obs_property_t *room_blend = obs_properties_add_float_slider(props, "room_blend", obs_module_text("RoomBlend"),
								      0.0, 1.0, 0.05);
	rvm_set_help(room_blend, "RoomBlend.Description");

	obs_property_t *use_plate = obs_properties_add_bool(props, "use_plate", obs_module_text("UsePlate"));
	rvm_set_help(use_plate, "UsePlate.Description");
	const struct rvm_filter *filter = data;
	char capture_text[128];
	if (filter && filter->countdown_label > 1)
		snprintf(capture_text, sizeof(capture_text), "%s %d", obs_module_text("CapturePlateCountdown"),
			 filter->countdown_label - 1);
	else if (filter && filter->countdown_label == 1)
		snprintf(capture_text, sizeof(capture_text), "%s", obs_module_text("CapturePlateRunning"));
	else if (filter && filter->countdown_label < 0)
		snprintf(capture_text, sizeof(capture_text), "%s", obs_module_text("CapturePlateFailed"));
	else
		snprintf(capture_text, sizeof(capture_text), "%s", obs_module_text("CapturePlate"));
	obs_property_t *capture =
		obs_properties_add_button2(props, "capture_plate", capture_text, rvm_capture_plate_clicked, data);
	rvm_set_help(capture, "CapturePlate.Description");
	obs_property_t *forget = obs_properties_add_button2(props, "forget_plate", obs_module_text("ForgetPlate"),
							    rvm_forget_plate_clicked, data);
	rvm_set_help(forget, "ForgetPlate.Description");
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
