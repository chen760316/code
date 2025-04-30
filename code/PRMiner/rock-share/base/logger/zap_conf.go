package logger

import (
	"os"
	"path"
	"strings"
	"time"

	"github.com/LinkinStars/golang-util/gu"
	"github.com/getsentry/sentry-go"
	rotatelogs "github.com/lestrrat-go/file-rotatelogs"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// yourProjectName 你的项目名称，用于命名打印日志名字和截短日志输出目录
var yourProjectName = "rock-backend"

// initZap 初始化zap日志配置
// projectName: 项目名称
// logPath: 日志打印目录
// maxAge: 日志最大存在时间，单位：天
// rotationTime: 日志切分时间，单位：小时
// rotationSize: 日志切分大小 单位：MB
func initZap(projectName, logPath string, maxAge, rotationTime time.Duration, rotationSize uint32, dsn string) {
	if len(projectName) != 0 {
		yourProjectName = projectName
	}

	maxAge = maxAge * 24 * time.Hour
	rotationTime = rotationTime * time.Hour
	if rotationSize == 0 {
		rotationSize = 1024 //1G
	}
	rotationSizeMB := int64(rotationSize * 1024 * 1024)
	// 创建日志存放目录
	if err := gu.CreateDirIfNotExist(logPath); err != nil {
		panic(err)
	}
	logPath = path.Join(logPath, projectName)

	// error日志文件配置
	errWriter, err := rotatelogs.New(
		logPath+"_err_%Y-%m-%d.log",
		rotatelogs.WithLinkName(logPath+"_err_last.log"), // 软链,指向最新日志文件
		rotatelogs.WithMaxAge(maxAge),
		rotatelogs.WithRotationTime(rotationTime),
		rotatelogs.WithRotationSize(rotationSizeMB),
	)
	if err != nil {
		panic(err)
	}

	// info日志文件配置
	infoWriter, err := rotatelogs.New(
		logPath+"_info_%Y-%m-%d.log",
		rotatelogs.WithLinkName(logPath+"_info_last.log"), // 软链,指向最新日志文件
		rotatelogs.WithMaxAge(maxAge),
		rotatelogs.WithRotationTime(rotationTime),
		rotatelogs.WithRotationSize(rotationSizeMB),
	)
	if err != nil {
		panic(err)
	}

	// 优先级设置
	highPriority := zap.LevelEnablerFunc(func(lvl zapcore.Level) bool {
		return lvl > zapcore.WarnLevel
	})
	lowPriority := zap.LevelEnablerFunc(func(lvl zapcore.Level) bool {
		return lvl >= zapcore.DebugLevel
	})

	// 控制台输出设置
	consoleDebugging := zapcore.Lock(os.Stdout)
	consoleEncoderConfig := zap.NewDevelopmentEncoderConfig()
	consoleEncoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
	consoleEncoderConfig.EncodeTime = timeEncoder
	consoleEncoderConfig.EncodeCaller = customCallerEncoder
	consoleEncoder := zapcore.NewConsoleEncoder(consoleEncoderConfig)

	// 文件输出设置
	errorCore := zapcore.AddSync(errWriter)
	infoCore := zapcore.AddSync(infoWriter)
	fileEncodeConfig := zap.NewProductionEncoderConfig()
	fileEncodeConfig.EncodeTime = timeEncoder
	fileEncodeConfig.EncodeCaller = customCallerEncoder
	fileEncoder := zapcore.NewJSONEncoder(fileEncodeConfig)

	core := zapcore.NewTee(
		zapcore.NewCore(fileEncoder, errorCore, highPriority),
		zapcore.NewCore(fileEncoder, infoCore, lowPriority),
		zapcore.NewCore(consoleEncoder, consoleDebugging, zapcore.DebugLevel),
	)

	// 显示行号
	caller := zap.AddCaller()

	development := zap.Development()
	logger := zap.New(core, caller, development)
	// 替换全局日志
	zap.ReplaceGlobals(logger)

	// 将系统输出重定向到zap中，保证所有出现异常均能打印到文件中
	if _, err := zap.RedirectStdLogAt(logger, zapcore.ErrorLevel); err != nil {
		panic(err)
	}
}

// customCallerEncoder 自定义打印路径，减少输出日志打印路径长度，根据输入项目名进行减少
func customCallerEncoder(caller zapcore.EntryCaller, enc zapcore.PrimitiveArrayEncoder) {
	str := caller.String()
	index := strings.Index(str, yourProjectName)
	if index == -1 {
		enc.AppendString(caller.FullPath())
	} else {
		index = index + len(yourProjectName) + 1
		enc.AppendString(str[index:])
	}
}

// timeEncoder 格式化日志时间，官方的不好看
func timeEncoder(t time.Time, enc zapcore.PrimitiveArrayEncoder) {
	enc.AppendString(t.Format("2006-01-02 15:04:05.000"))
}

// 将zap的Level转换为sentry的Level
func sentryLevel(lvl zapcore.Level) sentry.Level {
	switch lvl {
	case zapcore.DebugLevel:
		return sentry.LevelDebug
	case zapcore.InfoLevel:
		return sentry.LevelInfo
	case zapcore.WarnLevel:
		return sentry.LevelWarning
	case zapcore.ErrorLevel:
		return sentry.LevelError
	case zapcore.DPanicLevel:
		return sentry.LevelFatal
	case zapcore.PanicLevel:
		return sentry.LevelFatal
	case zapcore.FatalLevel:
		return sentry.LevelFatal
	default:
		return sentry.LevelFatal
	}
}

// SentryCoreConfig 定义 Sentry Core 的配置参数.
type SentryCoreConfig struct {
	Tags              map[string]string
	DisableStacktrace bool
	Level             zapcore.Level
	FlushTimeout      time.Duration
	Hub               *sentry.Hub
}

// sentryCore sentrycore的Core结构体，用于实现Core接口
type sentryCore struct {
	client               *sentry.Client    // sentry客户端
	cfg                  *SentryCoreConfig // core配置
	zapcore.LevelEnabler                   // LevelEnabler接口
	flushTimeout         time.Duration     // sentry上报的flush时间

	fields map[string]interface{} // 保存Fields
}

// With接口方法的实际实现，对传入fields进行设置日志打印时的打印解析方式并添加到已有的fields中
func (c *sentryCore) with(fs []zapcore.Field) *sentryCore {
	// Copy our map.
	m := make(map[string]interface{}, len(c.fields))
	for k, v := range c.fields {
		m[k] = v
	}

	// Add fields to an in-memory encoder.
	enc := zapcore.NewMapObjectEncoder()
	for _, f := range fs {
		f.AddTo(enc)
	}

	// Merge the two maps.
	for k, v := range enc.Fields {
		m[k] = v
	}

	return &sentryCore{
		client:       c.client,
		cfg:          c.cfg,
		fields:       m,
		LevelEnabler: c.LevelEnabler,
	}
}

// With 实现Core接口的With方法
func (c *sentryCore) With(fs []zapcore.Field) zapcore.Core {
	return c.with(fs)
}

// Check 实现Core接口的Check方法，只有大于在core配置中的的Level才会被打印
func (c *sentryCore) Check(ent zapcore.Entry, ce *zapcore.CheckedEntry) *zapcore.CheckedEntry {
	if c.cfg.Level.Enabled(ent.Level) {
		return ce.AddCore(ent, c)
	}
	return ce
}

// Write 实现Core接口的Write方法，对sentry进行上报，Fields作为Extra信息上报
func (c *sentryCore) Write(ent zapcore.Entry, fs []zapcore.Field) error {
	clone := c.with(fs)

	event := sentry.NewEvent()
	event.Message = ent.Message
	event.Timestamp = ent.Time
	event.Level = sentryLevel(ent.Level)
	event.Platform = "rock"
	event.Extra = clone.fields
	event.Tags = c.cfg.Tags

	if !c.cfg.DisableStacktrace {
		trace := sentry.NewStacktrace()
		if trace != nil {
			event.Exception = []sentry.Exception{{
				Type:       ent.Message,
				Value:      ent.Caller.TrimmedPath(),
				Stacktrace: trace,
			}}
		}
	}

	hub := c.cfg.Hub
	if hub == nil {
		hub = sentry.CurrentHub()
	}
	_ = c.client.CaptureEvent(event, nil, hub.Scope())

	if ent.Level > zapcore.ErrorLevel {
		c.client.Flush(c.flushTimeout)
	}
	return nil
}

// Sync 实现Core接口的Sync方法
func (c *sentryCore) Sync() error {
	c.client.Flush(c.flushTimeout)
	return nil
}

// NewSentryCore 生成Core对象
func NewSentryCore(cfg SentryCoreConfig, sentryClient *sentry.Client) zapcore.Core {

	core := sentryCore{
		client:       sentryClient,
		cfg:          &cfg,
		LevelEnabler: cfg.Level,
		flushTimeout: 3 * time.Second,
		fields:       make(map[string]interface{}),
	}

	if cfg.FlushTimeout > 0 {
		core.flushTimeout = cfg.FlushTimeout
	}

	return &core
}
