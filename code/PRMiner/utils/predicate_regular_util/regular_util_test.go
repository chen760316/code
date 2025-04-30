package regular_util

import (
	"context"
	"fmt"

	"net/http"
	_ "net/http/pprof"
	"testing"

	"github.com/bovinae/common/util"
	. "github.com/smartystreets/goconvey/convey"
)

func TestGetFieldTypeRegular(t *testing.T) {
	go func() {
		err := http.ListenAndServe(":8081", nil)
		if err != nil {
			fmt.Println("http.ListenAndServe failed, err:", err)
		}
	}()
	ru := NewRegularUtil()
	ctx := context.Background()
	Convey("TestGetFieldTypeRegular", t, func() {
		Convey("TestGetFieldTypeRegular", func() {
			data, err := util.NewCsvClient().ReadCsvFile(ctx, `./business_license_info_o_train.csv`)
			So(err, ShouldEqual, nil)
			for j := 0; j < len(data[0]); j++ {
				tmp := make([]any, 0, len(data))
				for i := 1; i < len(data); i++ {
					tmp = append(tmp, data[i][j])
				}
				predicates, rowNums, conflictRowNums := ru.GetRegularPredict(ctx, data[0][j], "string", tmp, 0.9)
				for _, predicate := range predicates {
					fmt.Printf("[%v: %v]\n", data[0][j], *predicate)
					fmt.Printf("rowNums: %v\n", rowNums[predicate.ConstantValue.(string)])
					fmt.Printf("conflictRowNums: %v\n", conflictRowNums[predicate.ConstantValue.(string)])
				}
			}
		})
	})
}

func TestDifferenceSet(t *testing.T) {
	a := []int32{2, 5}
	fmt.Println(differenceSet(10, a))
}
