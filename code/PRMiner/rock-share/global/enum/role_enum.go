package enum

const SYSTEM_ROLE_DESC = "系统角色"

const (
	PUBLISHED   = "PUBLISHED"
	UNPUBLISHED = "UNPUBLISHED"
)

const (
	CONTENT   = "CONTENT"
	TANDCNAME = "TANDCNAME"
)

const (
	SYSTEM     = "SYSTEM"
	NOT_SYSTEM = "NOT_SYSTEM"
	CATEGORY   = "自定义"
)

const (
	DETECT = "DETECT"
	HANDLE = "HANDLE"
)

const (
	DetectTypeFieldType       = "FIELDTYPE"
	DetectTypeLengthLimit     = "LENGTHLIMIT"
	DetectTypeContentRestrict = "CONTENTRESTRICT"
	DetectTypeEnumLimit       = "ENUMLIMIT"
	DetectTypeNumberLimit     = "NUMBERLIMIT"
	DetectTypeRegexLimit      = "REGEXLIMIT"
)

const (
	HandleTypeSpecialCharacterRemove = "SPECIALCHARACTERREMOVE"
	HandleTypeCharacterTransform     = "CHARACTERTRANSFORM"
	HandleTypeCharacterReplace       = "CHARACTERREPLACE"
	HandleTypeCharacterDelete        = "CHARACTERDELETE"
	HandleTypeValueMapping           = "VALUEMAPPING"
	HandleTypeSimilarValueMapping    = "SIMILARVALUEMAPPING"
)
