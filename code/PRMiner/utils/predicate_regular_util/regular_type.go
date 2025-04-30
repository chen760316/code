package regular_util

const (
	FC_CHINESE FieldCategory = iota // 只要包含中文就算作中文，包含特殊符号
	FC_NUMBER                       // 纯数字
	FC_ENGLISH                      // 英文数字算作英文，有特殊符号(除, . -外)就算作特殊符号类
	FC_OTHERS
	FC_TOTAL
)

type FieldCategory int8

func (fc FieldCategory) GetRegular() string {
	return fcPattern[fc]
}

const (
	// 静态字段类型
	FT_DATE_TIME FieldType = iota
	FT_RGB
	FT_CAR_NUMBER
	FT_PROVINCE
	FT_ADDRESS
	FT_ID_NUMBER
	FT_FIXED_TEL
	FT_PHONE
	FT_EMAIL
	FT_WEB_ADDR
	FT_ZIP_CODE
	FT_IP
	FT_MAC
	FT_UNIFIED_SOCIAL_CREDIT_NUMBER
	FT_CHINESE_NAME
	FT_ENGLISH_NAME
	FT_DOMAIN_NAME
	FT_SHOP_NAME
	FT_INTEGER // all integer
	FT_FLOAT   // has float

	// 动态字段类型
	// FT_CONTAIN
	FT_LENGTH
	FT_PREFIX
	FT_SUFFIX
	FT_ENUM
	// FT_SCOPE

	FT_TOTAL
)

type FieldType int16

var (
	FieldTypeName = map[FieldType]string{
		FT_DATE_TIME:                    "datetime",
		FT_RGB:                          "RGB",
		FT_CAR_NUMBER:                   "car number",
		FT_PROVINCE:                     "province",
		FT_ADDRESS:                      "address",
		FT_ID_NUMBER:                    "id number",
		FT_FIXED_TEL:                    "fixed telephone",
		FT_PHONE:                        "mobile phone",
		FT_EMAIL:                        "email",
		FT_WEB_ADDR:                     "web address",
		FT_ZIP_CODE:                     "zip code",
		FT_IP:                           "ip address",
		FT_MAC:                          "mac address",
		FT_UNIFIED_SOCIAL_CREDIT_NUMBER: "统一社会信用号",
		FT_CHINESE_NAME:                 "chinese name",
		FT_ENGLISH_NAME:                 "english name",
		FT_DOMAIN_NAME:                  "domain name",
		FT_SHOP_NAME:                    "shop name",
		FT_INTEGER:                      "integer",
		FT_FLOAT:                        "float",

		// FT_CONTAIN: "contain",
		FT_LENGTH: "length",
		FT_PREFIX: "prefix",
		FT_SUFFIX: "suffix",
		FT_ENUM:   "enum",
		// FT_SCOPE:   "scope",
	}
)

var (
	fcPattern = map[FieldCategory]string{
		FC_CHINESE: "^.*[\u4e00-\u9fa5]+.*$",
		FC_ENGLISH: `^([a-zA-Z]+[a-zA-Z\d]*[,\.\-!]?\s?)+$`,
		FC_NUMBER:  `^.?\d+$`,
	}

	ftPattern = map[FieldType]string{
		FT_DATE_TIME:                    `^\d{4}[年/-]{1}\d{1,2}[月/-]{1}\d{1,2}日?(\s+([0-1]?\d|2[0-3])[:-]{1}([0-5]\d)([:-]{1}[0-5]\d)?)?$`,
		FT_RGB:                          `^(#([0-9a-fA-F]{6}|[0-9a-fA-F]{3}))|([rR][gG][Bb][Aa]?\(((2[0-4]\d|25[0-5]|[01]?\d{2}?),\s?){2}(2[0-4]\d|25[0-5]|[01]?\d{2}?),?\s?(0\.\d{1,2}|1|0)?\){1})$`,
		FT_CAR_NUMBER:                   `^(([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-HJ-NP-Z](([0-9]{5}[ADF])|([ADF]([A-HJ-NP-Z0-9])[0-9]{4})))|([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-HJ-NP-Z][A-HJ-NP-Z0-9]{4}[A-HJ-NP-Z0-9挂学警港澳使领]))$`,
		FT_PROVINCE:                     `^(.+省|.+自治区|上海|北京|天津|重庆)市?$`,
		FT_ADDRESS:                      `^(.+县|.+市|.+镇|.+区|.+乡|.+场|.+海域|.+岛|.+街道){1}\s?(.+村)?(.+栋)?.*$`,
		FT_ID_NUMBER:                    `^(([1-6][1-9]|50)\d{4}(18|19|20)\d{2}((0[1-9])|10|11|12)(([0-2][1-9])|10|20|30|31)\d{3}[0-9Xx])|(([1-6][1-9]|50)\d{4}\d{2}((0[1-9])|10|11|12)(([0-2][1-9])|10|20|30|31)\d{3})$`,
		FT_FIXED_TEL:                    `^0\d{2,3}-[1-9]\d{6,7}$`,
		FT_PHONE:                        `^(13[0-9]|14[5|7]|15[0|1|2|3|5|6|7|8|9]|18[0|1|2|3|5|6|7|8|9])[0-9*]{4}\d{4}$`,
		FT_EMAIL:                        "^[\u4e00-\u9fa5A-Za-z0-9]+@[a-zA-Z0-9_-]+(\\.[a-zA-Z0-9_-]+)+$",
		FT_WEB_ADDR:                     `^(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?$`,
		FT_ZIP_CODE:                     `^[0-9]\d{5}$`,
		FT_IP:                           `^((\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5])\.){3}(\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5])$`,
		FT_MAC:                          `^[a-fA-F0-9]{2}([-:]?[a-fA-F0-9]{2})([-:.]?[a-fA-F0-9]{2})([-:]?[a-fA-F0-9]{2})([-:.]?[a-fA-F0-9]{2})([-:]?[a-fA-F0-9]{2})$`,
		FT_UNIFIED_SOCIAL_CREDIT_NUMBER: `^[0-9A-HJ-NPQRTUWXY]{2}\d{6}[0-9A-HJ-NPQRTUWXY]{10}$`,
		FT_CHINESE_NAME:                 "^[\u4e00-\u9fa5]{1}[\u4e00-\u9fa5*]{1,2}$",
		FT_ENGLISH_NAME:                 `^[A-Za-z]*(\s[A-Za-z]*)*$`,
		FT_DOMAIN_NAME:                  `^([a-zA-Z0-9]([a-zA-Z0-9-_]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,11}$`,
		FT_SHOP_NAME:                    `(.+店|.+服饰|.+公司|.+经营部|.+分店)`,
		FT_INTEGER:                      `^\d+$`,
		FT_FLOAT:                        `^\d*.?\d+$`,

		// FT_CONTAIN: `.*%s.*`,
		FT_LENGTH: `^.{%v}$`,
		FT_PREFIX: `^%v.*`,
		FT_SUFFIX: `.*%v$`,
		FT_ENUM:   `(%v)`,
		// FT_SCOPE:   `^.{%v}%s{%v}.*`,
	}
)

type FieldTypeRegular struct {
	FieldType FieldType
	Regular   string
}
