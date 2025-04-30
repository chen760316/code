package config

import (
	"fmt"
	"gopkg.in/yaml.v3"
	"os"

	"github.com/spf13/viper"
)

var AllStandConfig *StandardConfig
var RoleNameSet map[string]bool
var SimpleNameToFullName map[string]string
var ColumnNameToModelName map[string]string
var ChineseToEnglish map[string]string
var ChineseNameToEnglishName map[string]string
var ChineseNameToEnglishDesc map[string]string

func initStandardConfig() {
	v := viper.New()
	defaultPath := DefaultPath
	v.AddConfigPath(defaultPath)
	v.SetConfigName("config-standard-args")
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
	// 配置映射到结构体
	AllStandConfig = &StandardConfig{}

	if err := v.Unmarshal(AllStandConfig); err != nil {
		panic(err)
	}

	//jsonDataAllStandConfig, err := json.MarshalIndent(*AllStandConfig, "", "    ")
	//if err != nil {
	//	fmt.Println("JSON marshaling failed:", err)
	//	panic(err)
	//}
	//
	//fmt.Printf("config file config-standard-args.yml:\n%+v\n", string(jsonDataAllStandConfig))
	RoleNameSet = make(map[string]bool, 0)
	for roleName := range AllStandConfig.StandardArgConfig {
		RoleNameSet[roleName] = true
	}

	ChineseNameToEnglishName = make(map[string]string, 0)
	ChineseNameToEnglishDesc = make(map[string]string, 0)
	for roleName := range AllStandConfig.StandardArgConfig {
		for _, detect := range AllStandConfig.StandardArgConfig[roleName].Detection {
			ChineseNameToEnglishName[detect.Name] = detect.EnName
			ChineseNameToEnglishDesc[detect.Name] = detect.EnDesc
		}
	}
	ChineseNameToEnglishName["正则表达式检查"] = "Regular expression detection"
	ChineseNameToEnglishDesc["正则表达式检查"] = "Regular expression cannot match."
	ChineseNameToEnglishName["空值检查"] = "Null value detection"
	ChineseNameToEnglishDesc["空值检查"] = "No null value allowed."
	ChineseNameToEnglishName["枚举值检查"] = "Enumeration value detection"
	ChineseNameToEnglishDesc["枚举值检查"] = "Not in enumeration value."
	ChineseNameToEnglishName["数值检查"] = "Number check"
	ChineseNameToEnglishDesc["数值检查"] = "The number does not meet the requirements."
	ChineseNameToEnglishName["长度检查"] = "Length check"
	ChineseNameToEnglishDesc["长度检查"] = "The length does not meet the requirements."

	ChineseToEnglish = make(map[string]string, 0)
	// 读取YAML文件
	yamlFile2, err := os.ReadFile("config/chinese-english.yml")
	if err != nil {
		fmt.Println("ReadFile config-school.yml failed:", err)
		panic(err)
	}
	// 解析YAML文件内容
	err = yaml.Unmarshal(yamlFile2, &ChineseToEnglish)
	for roleName := range AllStandConfig.StandardArgConfig {
		for _, detect := range AllStandConfig.StandardArgConfig[roleName].Detection {
			ChineseToEnglish[detect.Desc] = detect.EnDesc
		}
	}
	ColumnNameToModelName = make(map[string]string, 0)
	// 读取YAML文件
	yamlFile1, err := os.ReadFile("config/config-model-name.yml")
	if err != nil {
		fmt.Println("ReadFile config-school.yml failed:", err)
		panic(err)
	}
	// 解析YAML文件内容
	err = yaml.Unmarshal(yamlFile1, &ColumnNameToModelName)

	SimpleNameToFullName = make(map[string]string, 0)
	// 读取YAML文件
	yamlFile, err := os.ReadFile("config/config-school.yml")
	if err != nil {
		fmt.Println("ReadFile config-school.yml failed:", err)
		panic(err)
	}
	// 解析YAML文件内容
	var data map[string][]string
	err = yaml.Unmarshal(yamlFile, &data)
	if err != nil {
		fmt.Println("映射数据 failed:", err)
		panic(err)
	}
	for k, values := range data {
		for i := 0; i < len(values); i++ {
			_, ok := SimpleNameToFullName[values[i]]
			if ok {
				SimpleNameToFullName[values[i]] = ""
			} else {
				SimpleNameToFullName[values[i]] = k
			}
		}
	}
}

type StandOperationRule struct {
	Neatness              []Rule `yaml:"Neatness"`
	Detection             []Rule `yaml:"Detection"`
	Enhancement           []Rule `yaml:"Enhancement"`
	InformationExtraction []Rule `yaml:"InformationExtraction"`
}

type Rule struct {
	Type      string  `yaml:"type"` //similarity相似度、regex正则、function函数
	Arg       any     `yaml:"arg"`
	Name      string  `yaml:"name"`
	Desc      string  `yaml:"desc"`
	Dimension string  `yaml:"dimension"`
	Err       string  `yaml:"err"`       //没匹配时，错误提示语句
	Threshold float64 `yaml:"threshold"` //角色阈值
	EnDesc    string  `yaml:"enDesc"`
	EnName    string  `yaml:"enName"`
}

type StandardSimilarityConfig struct {
	Threshold float64 `yaml:"threshold"`
	Distance  float64 `yaml:"distance"`
}

type RoleRuleConfig struct {
	SimilarThreshold  float64 `yaml:"similarThreshold"`
	IdentifyThreshold float64 `yaml:"identifyThreshold"`
}

type StandardArgConfig map[string]StandOperationRule

type StandardConfig struct {
	StandardSimilarityConfig StandardSimilarityConfig `yaml:"standardSimilarityConfig"`
	StandardArgConfig        StandardArgConfig        `yaml:"standardArgConfig"`
	RoleRuleConfig           RoleRuleConfig           `yaml:"roleRuleConfig"`
}
