package utils

// func ChoseDataOrigin(requestJson *ruleFinderRequest, gv *global_variables.GlobalV) map[string]int {
// skipColumn := map[string]int{"id": 1, "row_id": 1, "update_time": 1}
// gv.TableName =
// switch dataOrigin {
// case 1:
// 	gv.TableName = "relation_scy"
// 	gv.CR = 0.05
// 	gv.FTR = 0.85
// 	rds_go.DB, _ = sql.Open("mysql", "root:PoD2020@sics@tcp(192.168.13.93:3306)/testdata")
// 	rds_go.DB.SetConnMaxLifetime(100)
// 	rds_go.DB.SetMaxIdleConns(10)
// case 2:
// 	gv.TableName = "tax_4w"
// 	gv.CR = 0.0000001
// 	gv.FTR = 0.5
// 	rds_go.DB, _ = sql.Open("mysql", "root:PoD2020@sics@tcp(192.168.13.93:3306)/testdata")
// 	rds_go.DB.SetConnMaxLifetime(100)
// 	rds_go.DB.SetMaxIdleConns(10)
// case 3:
// 	gv.TableName = "tax_1w"
// 	gv.CR = 0.0000001
// 	gv.FTR = 0.5
// 	rds_go.DB, _ = sql.Open("mysql", "root:PoD2020@sics@tcp(192.168.13.93:3306)/testdata")
// 	rds_go.DB.SetConnMaxLifetime(100)
// 	rds_go.DB.SetMaxIdleConns(10)
// case 4:
// 	gv.TableName = "jili_100"
// 	gv.CR = 0.3
// 	gv.FTR = 0.9
// 	rds_go.DB, _ = sql.Open("mysql", "root:PoD2020@sics@tcp(192.168.13.93:3306)/testdata")
// 	rds_go.DB.SetConnMaxLifetime(100)
// 	rds_go.DB.SetMaxIdleConns(10)
// case 5:
// 	//建表时,XSFMC、GMFMC、BZ列没有导入
// 	gv.TableName = "guoshui_100"
// 	skipColumn = map[string]int{"KPRQ": 1, "JSSJ": 1, "RKSJ": 1, "FPZT_YWSJ": 1, "FPZT_WHSJ": 1, "SJQFRQ": 1, "ZJQDRQ": 1, "TB_SJ": 1}
// 	gv.CR = 0.05
// 	gv.FTR = 0.85
// 	rds_go.DB, _ = sql.Open("mysql", "root:PoD2020@sics@tcp(192.168.13.93:3306)/testdata")
// 	rds_go.DB.SetConnMaxLifetime(100)
// 	rds_go.DB.SetMaxIdleConns(10)
// case 6:
// 	//建表时,XSFMC、GMFMC、BZ列没有导入
// 	gv.TableName = "guoshui_10000"
// 	skipColumn = map[string]int{"KPRQ": 1, "JSSJ": 1, "RKSJ": 1, "FPZT_YWSJ": 1, "FPZT_WHSJ": 1, "SJQFRQ": 1, "ZJQDRQ": 1, "TB_SJ": 1}
// 	gv.CR = 0.05
// 	gv.FTR = 0.85
// 	rds_go.DB, _ = sql.Open("mysql", "root:PoD2020@sics@tcp(192.168.13.93:3306)/testdata")
// 	rds_go.DB.SetConnMaxLifetime(100)
// 	rds_go.DB.SetMaxIdleConns(10)
// case 7:
// 	// nice
// 	//建表时,XSFMC、GMFMC、BZ列没有导入
// 	gv.TableName = "guoshui_5000"
// 	skipColumn = map[string]int{"KPRQ": 1, "JSSJ": 1, "RKSJ": 1, "FPZT_YWSJ": 1, "FPZT_WHSJ": 1, "SJQFRQ": 1, "ZJQDRQ": 1, "TB_SJ": 1}
// 	gv.CR = 0.05
// 	gv.FTR = 0.85
// 	rds_go.DB, _ = sql.Open("mysql", "root:PoD2020@sics@tcp(192.168.13.93:3306)/testdata")
// 	rds_go.DB.SetConnMaxLifetime(100)
// 	rds_go.DB.SetMaxIdleConns(10)
// case 8:
// 	gv.TableName = "tax_1w_copy1"
// 	gv.CR = 0.0000001
// 	gv.FTR = 0.5
// 	rds_go.DB, _ = sql.Open("mysql", "root:PoD2020@sics@tcp(192.168.13.93:3306)/testdata")
// 	rds_go.DB.SetConnMaxLifetime(100)
// 	rds_go.DB.SetMaxIdleConns(10)
// }
// 	return skipColumn
// }
