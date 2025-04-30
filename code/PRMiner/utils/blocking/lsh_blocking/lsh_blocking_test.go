package lsh_blocking

import (
	"fmt"
	"rds-shenglin/storage/config/storage_initiator"
	"os"
	"testing"
)

func Test_doBlocking(t *testing.T) {
	list := doBlocking([][]int{
		{100, 1, 2, 3, 10, 11, 12},
		{200, 4, 5, 6, 21, 22, 23},
		{300, 1, 5, 3, 32, 33, 34},
	})
	for _, tokenIds := range list {
		fmt.Printf("%v\n", tokenIds)

	}
}

func TestBlocking(t *testing.T) {

	var taskId int64 = 100123
	blkId := 5
	tableName := "tt"
	columnName := "cc"
	vectorsFile := "f.txt"

	err := os.WriteFile(vectorsFile, []byte("100, 1, 2, 3, 10, 11, 12\n200, 4, 5, 6, 21, 22, 23\n300, 1, 5, 3, 32, 33, 34"), 0666)
	if err != nil {
		panic(err)
	}

	storage_initiator.InitStorage()
	leftVectorList, rightVectorList, err := Blocking(taskId, blkId, tableName, columnName, vectorsFile, tableName, columnName, vectorsFile)
	if err != nil {
		panic(err)
	}
	fmt.Println(leftVectorList)
	fmt.Println(rightVectorList)
}
