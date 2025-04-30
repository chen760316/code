package rds

import (
	"encoding/json"
)

type ConnectKey struct {
	LeftTableId       string   `json:"leftTableId"`
	RightTableId      string   `json:"rightTableId"`
	LeftTableColumns  []string `json:"leftTableColumns"`  //目前单一
	RightTableColumns []string `json:"rightTableColumns"` //目前单一
}

type RDSRequest struct {
	TablesID         []string               `json:"tablesID" binding:"required"`
	TaskID           int64                  `json:"taskID" binding:"required"`
	JoinKeys         []ConnectKey           `json:"joinKeys"`         // 主外键
	OtherMappingKeys []ConnectKey           `json:"otherMappingKeys"` // 非主外键的跨列谓词
	UDFTabCols       []UDFTabCol            `json:"UDFTabCols"`       // 传不了
	SkipYColumns     []string               `json:"skipYColumns"`     // 作为Y的列
	SkipColumns      []string               `json:"skipColumns"`      // 不参与规则发现的列
	MutexGroup       [][]string             `json:"mutexGroup"`       // 一组中只有一列可以出现在规则中
	Eids             map[string]string      `json:"eids"`             // 实体列id
	Conf             map[string]interface{} `json:"conf"`             //暂时不传
	DefinedRoleRules []Rule                 `json:"definedRoleRules"` // 用户自定义角色规则
}

type UDFTabCol struct {
	LeftTableId               string      `json:"leftTableId,omitempty"`
	LeftColumnName            string      `json:"leftColumnName,omitempty"`
	RightTableId              string      `json:"rightTableId,omitempty"`              // 左右列相同时，不传即可
	RightColumnName           string      `json:"rightColumnName,omitempty"`           // 左右列相同时，不传即可
	Type                      string      `json:"type,omitempty"`                      // similar/ML
	Name                      string      `json:"name,omitempty"`                      // 比如 jaccard sentence-bert
	Threshold                 float64     `json:"threshold,omitempty"`                 // 阈值，只在相似度时有用
	LeftColumnVectorFilePath  string      `json:"leftColumnVectorFilePath,omitempty"`  // 向量文件地址。只有 ML 时才有用
	RightColumnVectorFilePath string      `json:"rightColumnVectorFilePath,omitempty"` // 向量文件地址。左右列相同时，不传即可
	LeftVectorList            [][]float64 `json:"-"`                                   // 从 LeftColumnVectorFilePath 读取
	RightVectorList           [][]float64 `json:"-"`                                   // 从 RightColumnVectorFilePath ? LeftColumnVectorFilePath 读取
}

type UDFTabColKey string

func (udf *UDFTabCol) Key() UDFTabColKey {
	if udf.RightTableId == "" {
		udf.RightTableId = udf.LeftTableId
	}
	if udf.RightColumnName == "" {
		udf.RightColumnName = udf.LeftColumnName
	}
	if udf.RightColumnVectorFilePath == "" {
		udf.RightColumnVectorFilePath = udf.LeftColumnVectorFilePath
	}

	return UDFTabColKey(udf.String())
}

func (udf *UDFTabCol) String() string {
	bytes, err := json.Marshal(udf)
	if err != nil {
		return err.Error()
	}
	return string(bytes)
}
