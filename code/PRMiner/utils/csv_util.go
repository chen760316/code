package utils

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

func GetCsvCls(path string, skipColumn map[string]int) (map[string]int, int, error) {
	cls := make(map[string]int)
	var totalLine = 0
	preData, err := GetCsvData(path)
	if err != nil {
		fmt.Println("read a csv failed, err:", err)
		return nil, 0, ErrReadCsv
	}
	index := 0
	for _, columnName := range preData[0] {
		if _, ok := cls[columnName]; !ok {
			if _, okk := skipColumn[columnName]; !okk {
				cls[columnName] = index
			}
		}
		index++
	}
	totalLine = len(preData) - 1
	return cls, totalLine, nil
}

func GetCsvData(path string) ([][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		fmt.Println("opens a csv failed, err:", err)
		return nil, ErrOpenCsv
	}
	reader := csv.NewReader(f)
	preData, err := reader.ReadAll()
	if err != nil {
		fmt.Println("read a csv failed, err:", err)
		return nil, ErrReadCsv
	}
	return preData, nil
}

func CreateCsv(path string, data [][]string) error {
	csvFile, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer csvFile.Close()
	csvWriter := csv.NewWriter(csvFile)
	err = csvWriter.WriteAll(data)
	if err != nil {
		fmt.Printf("error (%v)", err)
		return err
	}
	return nil
}

func ReadCSVToMap(filePath string) (map[string][]interface{}, int, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, 0, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	headers, records, err := readCSV(reader)
	if err != nil {
		return nil, 0, err
	}

	result := make(map[string][]interface{})
	for _, header := range headers {
		result[header] = make([]interface{}, len(records))
	}

	for i, record := range records {
		for j, header := range headers {
			result[header][i] = parseValue(record[j])
		}
	}

	return result, len(records), nil
}

// 读取 CSV 内容并返回表头和记录
func readCSV(reader *csv.Reader) ([]string, [][]string, error) {
	headers, err := reader.Read()
	if err != nil {
		return nil, nil, err
	}

	var records [][]string
	for {
		record, err := reader.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			return nil, nil, err
		}
		records = append(records, record)
	}

	return headers, records, nil
}

// 将字符串值解析为合适的类型
func parseValue(value string) interface{} {
	// 尝试解析为整数
	if i, err := strconv.Atoi(value); err == nil {
		return i
	}

	// 尝试解析为浮点数
	if f, err := strconv.ParseFloat(value, 64); err == nil {
		return f
	}

	// 保留为字符串
	return value
}
