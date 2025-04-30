package dataManager

import (
	"encoding/csv"
	"io"
	"os"
	"rds-shenglin/rock-share/base/logger"
	"strconv"
)

// 定义一个 MyError 的结构体
type MyError struct {
	Msg string
}

// 实现 error 接口的 Error 方法
func (m *MyError) Error() string {
	return m.Msg
}

func readDataFromCsv(path string) ([][]float64, []string, string) {
	xList := make([]string, 0)
	data := make([][]float64, 0)
	fs, err := os.Open(path)
	if err != nil {
		logger.Errorf("can not open the file, err is %s", err.Error())
		panic(err)
	}
	defer fs.Close()
	r := csv.NewReader(fs)
	lineNum := 0
	for {
		row, err := r.Read()
		if err != nil && err != io.EOF {
			logger.Errorf("can not read, err is %s", err.Error())
			panic(err)
		}
		if err == io.EOF {
			break
		}
		//读取Attribute
		if lineNum == 0 {
			for i := 0; i < cap(row); i++ {
				xList = append(xList, row[i])
			}
		} else { //读取tuple
			rowData := make([]string, 0)
			for i := 0; i < cap(row); i++ {
				rowData = append(rowData, row[i])
			}
			floatData := make([]float64, len(rowData))
			for i := 0; i < len(rowData); i++ {
				floatData[i], err = strconv.ParseFloat(rowData[i], 64)
				if err != nil {
					logger.Errorf("string %s convert to float64 error : %s", rowData[i], err.Error())
					panic(err)
				}
			}
			data = append(data, floatData)
		}
		lineNum++
	}
	y := xList[len(xList)-1]
	xList = xList[:len(xList)-1]
	return data, xList, y
}
