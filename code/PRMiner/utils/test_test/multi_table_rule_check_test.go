package test_test

import (
	"rds-shenglin/rock-share/base/config"
	"rds-shenglin/rock-share/base/logger"
	"rds-shenglin/rock-share/global/model/rds"
	"rds-shenglin/local_version"
	"rds-shenglin/storage/storage2/database/database_facade/rich_dababase"
	"rds-shenglin/storage/storage2/database/database_facade/rich_dababase/irich_dababase"
	"rds-shenglin/storage/storage2/database/etl/extern_db/postgres"
	"rds-shenglin/storage/storage2/database/etl/import_from_DB/import_from_DB_config"
	"rds-shenglin/storage/storage2/database/table/table_impl/row_id"
	"rds-shenglin/storage/storage2/rock_db"
	"rds-shenglin/storage/storage2/storage3/database/database_impl/database_impl_s3_base"
	"rds-shenglin/storage/storage2/storage3/database/database_impl/database_impl_s3_tx/database_impl_s3_tx_impl"
	"rds-shenglin/storage/storage2/storage3/file_system/memory_file_system"
	"rds-shenglin/storage/storage2/storage3/page_memory/page_memory"
	"rds-shenglin/storage/storage2/utils"
	"rds-shenglin/storage/storage2/utils/test"
	"testing"
	"time"
)

// TestMultiTableRuleCheckExecute 多表查错测试
func TestMultiTableRuleCheckExecute(t *testing.T) {
	config.InitConfig()
	all := config.All
	l := all.Logger
	s := all.Server
	logger.InitLogger(l.Level, "rock", l.Path, l.MaxAge, l.RotationTime, l.RotationSize, s.SentryDsn)
	useDBS3(func(richDB irich_dababase.IRichDatabase) {
		rock_db.DB = richDB
		tab_delivery_note := "tab_delivery_note"
		tab_delivery_stop := "tab_delivery_stop"
		dataSourceName := postgres.DataSourceName("192.168.15.204", "5432", "rock", "rockPoD@2020", "rock_paper", "sslmode", "disable")
		pg, err := postgres.Open(dataSourceName)
		test.PanicErr(err)
		var importLength int64 = 0
		test.PanicErr(rock_db.DB.ImportFromDB(&import_from_DB_config.Config{
			ExternDataBase: pg,
			DBTableName:    "small1.tab_delivery_note",
			NewTableName:   tab_delivery_note,
			ImportLength:   importLength,
			ColumnConfigMap: map[string]*import_from_DB_config.ColumnConfig{
				"delivery_note_name": {DBColumnName: "delivery_note_name"},
				"shipping_address":   {DBColumnName: "shipping_address"},
			},
		}))
		test.PanicErr(rock_db.DB.ImportFromDB(&import_from_DB_config.Config{
			ExternDataBase: pg,
			DBTableName:    "small1.tab_delivery_stop",
			NewTableName:   tab_delivery_stop,
			ImportLength:   importLength,
			ColumnConfigMap: map[string]*import_from_DB_config.ColumnConfig{
				"delivery_note_name": {DBColumnName: "delivery_note_name"},
				"delivery_address":   {DBColumnName: "delivery_address"},
			},
		}))
		logger.Info(rock_db.DB.Show(tab_delivery_note))
		logger.Info(rock_db.DB.Show(tab_delivery_stop))
		f1 := rds.Predicate{
			PredicateStr: "t0.delivery_note_name = t2.delivery_note_name",
			LeftColumn: rds.Column{
				TableId:     tab_delivery_note,
				ColumnId:    "delivery_note_name",
				ColumnIndex: 0,
			},
			RightColumn: rds.Column{
				TableId:     tab_delivery_stop,
				ColumnId:    "delivery_note_name",
				ColumnIndex: 2,
			},
			ConstantValue: nil,
			PredicateType: 2,
		}
		f2 := rds.Predicate{
			PredicateStr: "t1.delivery_note_name = t3.delivery_note_name",
			LeftColumn: rds.Column{
				TableId:     tab_delivery_note,
				ColumnId:    "delivery_note_name",
				ColumnIndex: 1,
			},
			RightColumn: rds.Column{
				TableId:     tab_delivery_stop,
				ColumnId:    "delivery_note_name",
				ColumnIndex: 3,
			},
			ConstantValue: nil,
			PredicateType: 2,
		}
		p1 := rds.Predicate{
			PredicateStr: "t2.delivery_address = t3.delivery_address",
			LeftColumn: rds.Column{
				TableId:     tab_delivery_stop,
				ColumnId:    "delivery_address",
				ColumnIndex: 2,
			},
			RightColumn: rds.Column{
				TableId:     tab_delivery_stop,
				ColumnId:    "delivery_address",
				ColumnIndex: 3,
			},
			ConstantValue: nil,
			PredicateType: 1,
		}
		p2 := rds.Predicate{
			PredicateStr: "t0.shipping_address = t1.shipping_address",
			LeftColumn: rds.Column{
				TableId:     tab_delivery_note,
				ColumnId:    "shipping_address",
				ColumnIndex: 0,
			},
			RightColumn: rds.Column{
				TableId:     tab_delivery_note,
				ColumnId:    "shipping_address",
				ColumnIndex: 1,
			},
			ConstantValue: nil,
			PredicateType: 1,
		}
		rowIdPred := rds.Predicate{
			PredicateStr: "t0.row_id = 1",
			LeftColumn: rds.Column{
				TableId:     tab_delivery_note,
				ColumnId:    row_id.ColumnName,
				ColumnIndex: 0,
			},
			ConstantValue: int64(1),
			PredicateType: 0,
		}

		// 数据准备
		const taskId = 123456
		gv := local_version.InitGlobalV(taskId, []string{tab_delivery_note, tab_delivery_stop}, nil, nil, nil, nil)
		start := time.Now()
		rs, xs, xys, _ := local_version.CalRule([]rds.Predicate{f1, f2, p1, rowIdPred}, p2, gv)
		logger.Infof("%s %d %d %d\n", utils.DurationString(start), rs, xs, xys)
	})
}

func useDBS3(use func(richDB irich_dababase.IRichDatabase)) {
	mfs := memory_file_system.New()
	pageMemory := test.PanicErr1(page_memory.New(mfs, 16*1024, 5*1024*1024*1024))
	db3base := database_impl_s3_base.New(pageMemory, 10)
	db3tx := database_impl_s3_tx_impl.New(db3base)
	rdb := rich_dababase.New(db3tx)
	test.PanicErr(database_impl_s3_tx_impl.InitLogTables(db3tx))
	use(rdb)
}
