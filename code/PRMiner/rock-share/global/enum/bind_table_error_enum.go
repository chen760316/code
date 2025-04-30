package enum

type BindTableError int

const (
	FieldNameRepeat  string = "FIELD_NAME_REPEAT" //新建接口，检测此异常
	FieldBlank       string = "FIELD_BLANK"       //新建接口,可能插入此异常。配置接口，检测此异常
	FieldMatched     string = "FIELD_MATCHED"     //配置接口，检测此异常
	FieldDelete      string = "FIELD_DELETE"      //配置接口，检测此异常，只做删除操作
	FieldBindRepeat  string = "FIELD_BIND_REPEAT" //配置接口，检测此异常
	FieldUpdate      string = "FIELD_UPDATE"
	FieldTypeMatched string = "FIELD_TYPE_MATCHED"

	SourceTableNotFound        string = "SOURCE_TABLE_NOT_FOUND"
	SourceTableColumnNotFound  string = "SOURCE_TABLE_COLUMN_NOT_FOUND"
	TableColumnNotSupportError        = "TABLE_COLUMN_NOT_SUPPORT_ERROR"
	SourceTableFieldNotFound   string = "SOURCE_TABLE_FIELD_NOT_FOUND"
	ChildrenTableNotFound      string = "CHILDREN_TABLE_NOT_FOUND"
	MappingTableNotFound       string = "MAPPING_TABLE_NOT_FOUND"
)

const (
	FieldNameRepeatDesc  string = "关联的左右表存在重复列名"      //左表和右表有重复的列名
	FieldBlankDesc       string = "关联字段缺失"            //公共列未配置
	FieldMatchedDesc     string = "请确认填写完整或删除不需要的关联表" //公共列左右数量不匹配
	FieldDeleteDesc      string = "原关联字段已被删除"
	FieldBindRepeatDesc  string = "关联字段重复" //单表的某个列有被匹配多次
	FieldUpdateDesc      string = "原关联字段已修改列名"
	FieldTypeMatchedDesc string = "关联字段类型不匹配"

	SourceTableNotFoundDesc string = "找不到数据表"
	//SourceTableColumnNotFoundDesc string = "数据表为空表"
	//TableColumnNotSupportDesc     string = "存在系统不支持的数据类型"
	SourceTableFieldNotFoundDesc string = "找不到数据表属性"
	ChildrenTableNotFoundDesc    string = "上游表异常"
	MappingTableNotFoundDesc     string = "字段绑定的映射表丢失"
)
