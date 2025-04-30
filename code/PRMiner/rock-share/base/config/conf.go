package config

import (
	"fmt"
	"log"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/spf13/viper"
)

// All 全部配置索引
var All *AllConfig

var DefaultPath = "./config"
var DebugPath = "./base/config"

// InitConfig 初始化读取配置文件
func InitConfig() {
	// config.yml
	initConfig1()
	// config-standard-args.yml
	initStandardConfig()
}

// InitConfig 初始化读取配置文件
func initConfig1() {
	v := viper.New()
	//项目运行主目录
	//appPath := GetAppPath()
	//默认配置文件所在目录
	defaultPath := DefaultPath

	v.AddConfigPath(defaultPath)
	v.SetConfigName("config")
	configType := "yml"
	v.SetConfigType(configType)

	// 读取配置
	if err := v.ReadInConfig(); err != nil {
		panic(err)
	}

	configs := v.AllSettings()

	// SetDefault使用：全部以默认配置写入
	for k, val := range configs {
		v.SetDefault(k, val)
	}

	//增量配置
	debugEnv := os.Getenv("DEBUG")
	// 根据配置的env读取相应的配置信息
	if debugEnv == "true" {

		fmt.Println("debugEnv DEBUG=true")
		newPath := DebugPath
		debug := "debug"
		newConfigName := debug + ".yml"
		newConfigPath := newPath + "/" + newConfigName
		exists, _ := isExists(newConfigPath)

		if exists {
			fmt.Printf("%s exists\n", newConfigPath)
			v.AddConfigPath(newPath)
			v.SetConfigName(debug)
			v.SetConfigType(configType)
			err := v.ReadInConfig()
			if err != nil {
				panic(err)
			}
		} else {
			fmt.Printf("%s not exists\n", newConfigPath)
		}
	}

	// 监控配置文件变化并热加载程序
	v.WatchConfig()
	v.OnConfigChange(func(e fsnotify.Event) {
		log.Printf("Config file changed: %s", e.Name)
	})

	// 配置映射到结构体
	All = &AllConfig{}
	if err := v.Unmarshal(All); err != nil {
		panic(err)
	}

	if All.Storage.PagePoolSizeGB == 0 {
		All.Storage.PagePoolSizeGB = 5
	}
	if All.Storage.DataMaxLength == 0 {
		All.Storage.DataMaxLength = 1000000
	}
	if All.Storage.BindMaxLength == 0 {
		All.Storage.BindMaxLength = 5000
	}

	// 这里可以做检查，如果配置文件相关配置项异常亦可以不启动
	fmt.Printf("config file content:\n%+v\n", *All)

}

type Config struct {
	Name string
}

// AllConfig 全部配置文件
type AllConfig struct {
	Server     ServerConfig     `mapstructure:"server_config"`
	Logger     LoggerConfig     `mapstructure:"logger_config"`
	Pg         PgConfig         `mapstructure:"pg_config"`
	Statistics StatisticsConfig `mapstructure:"statistics_config"`
	Etcd       EtcdConfig       `mapstructure:"etcd_config"`
	Storage    StorageConfig    `mapstructure:"storage_config"`
	Other      OtherConfig      `mapstructure:"other_config"`
}

type PgConfig struct {
	Host         string `mapstructure:"host"`
	User         string `mapstructure:"user"`
	Password     string `mapstructure:"password"`
	DB           string `mapstructure:"dbname"`
	Port         uint32 `mapstructure:"port"`
	MaxOpenConns int    `mapstructure:"max_open_conns"`
	MaxIdleConns int    `mapstructure:"max_idle_conns"`
}

// ServerConfig 服务配置
type ServerConfig struct {
	HttpPort    string `mapstructure:"http_port"`
	SentryDsn   string `mapstructure:"sentry_dsn"`
	MaxMemory   int    `mapstructure:"max_memory"`
	MaxDateSize int    `mapstructure:"max_data_size"`
}

// LoggerConfig 日志配置
type LoggerConfig struct {
	Level        string        `mapstructure:"level"`
	Path         string        `mapstructure:"path"`
	MaxAge       time.Duration `mapstructure:"max_age"`
	RotationTime time.Duration `mapstructure:"rotation_time"`
	RotationSize uint32        `mapstructure:"rotation_size"`
}

type StatisticsConfig struct {
	Cardinality uint32  `mapstructure:"cardinality"`
	Percentage  float64 `mapstructure:"percentage"`
}

type EtcdConfig struct {
	Endpoints           []string `mapstructure:"endpoints"`
	DialTimeout         int32    `mapstructure:"dial_timeout"`
	Username            string   `mapstructure:"username"`
	Password            string   `mapstructure:"password"`
	ServiceDiscoveryKey string   `mapstructure:"service_discovery_key"`
}

type StorageConfig struct {
	PagePoolSizeGB int `mapstructure:"page_pool_size_gb"`
	DataMaxLength  int `mapstructure:"data_max_length"`
	BindMaxLength  int `mapstructure:"bind_max_length"`
}

type OtherConfig struct {
	PKFKRowLimit int `mapstructure:"pk_fk_row_limit"`
}

// GetAppPath 获取项目运行时的绝对目录
func GetAppPath() string {
	return getCurrentAbPath()
}

// 获取绝对路径。。最终方案-全兼容
func getCurrentAbPath() string {
	dir := getCurrentAbPathByExecutable()
	tmpDir, _ := filepath.EvalSymlinks(os.TempDir())
	if strings.Contains(dir, tmpDir) {
		return getCurrentAbPathByCaller()
	}
	return dir
}

// 获取当前执行文件绝对路径
func getCurrentAbPathByExecutable() string {
	exePath, err := os.Executable()
	if err != nil {
		log.Fatal(err)
	}
	res, _ := filepath.EvalSymlinks(filepath.Dir(exePath))
	return res
}

// 获取当前执行文件绝对路径（go run）
func getCurrentAbPathByCaller() string {
	var abPath string
	_, filename, _, ok := runtime.Caller(0)
	if ok {
		abPath = path.Dir(filename)
	}
	return abPath
}

// 判断所给文件/文件夹是否存在
func isExists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	}
	//isnotexist来判断，是不是不存在的错误
	if os.IsNotExist(err) { //如果返回的错误类型使用os.isNotExist()判断为true，说明文件或者文件夹不存在
		return false, nil
	}
	return false, err //如果有错误了，但是不是不存在的错误，所以把这个错误原封不动的返回
}
