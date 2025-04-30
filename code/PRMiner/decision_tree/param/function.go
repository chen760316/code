package conf

import (
	"rds-shenglin/decision_tree/param/conf_cluster"
	"rds-shenglin/decision_tree/param/conf_decisiontree"
	"rds-shenglin/decision_tree/param/conf_manager"
	"strings"
)

func Init() {
	conf_cluster.Init()
	conf_decisiontree.Init()
	GetFlagsOrder()
}

func ParseTaskArgs(args []string) error {
	err := conf_manager.ParseFlagsWithArgs(args)
	//打印命令行
	ArgsPrint()
	return err
}

func CurArgsToString() string {
	builder := strings.Builder{}
	builder.WriteString("cmd settings!!!\n===============================\n")
	builder.WriteString(conf_manager.FlagsToString())
	builder.WriteString("\n===============================\n")
	return builder.String()
}

func GetFlagsOrder() {
	conf_manager.FlagsPrintPriority()
}

func ArgsPrint() {
	conf_manager.FlagsPrint()
}
