package db_util

import (
	"fmt"
	"rds-shenglin/rock-share/base/logger"
	"rds-shenglin/rock-share/global/enum"
	"rds-shenglin/rds_config"
	"rds-shenglin/storage/specific_business/rule_table"
	"gorm.io/gorm"
	"strconv"
	"strings"

	"rds-shenglin/rock-share/global/db"
	"rds-shenglin/rock-share/global/model/po"
	"rds-shenglin/rock-share/global/model/rds"
	"rds-shenglin/utils"
)

func WriteRule(taskId string, tableId string, data rds.Rule) int {
	taskIdi, _ := strconv.ParseInt(taskId, 10, 64)
	tableIdi, _ := strconv.ParseInt(tableId, 10, 64)
	pgRule := po.Rule{
		Id:            0,
		TaskId:        taskIdi,
		ConflictCount: 0,
		Ree:           data.Ree,
		ReeJson:       utils.GetRuleJson(data),
		ReeType:       data.Rhs.PredicateType,
		Status:        enum.RULE_UNUSED,
	}
	err := po.CreateRule(&pgRule, db.DB)
	if err != nil {
		fmt.Println(err)
		return 0
	}
	ruleId := int(pgRule.Id)

	var ruleColumns []po.RuleColumn
	for _, column := range data.LhsColumns {
		columnId, _ := strconv.ParseInt(column.ColumnId, 10, 64)
		ruleColumn := po.RuleColumn{
			RuleId:   pgRule.Id,
			BindId:   tableIdi,
			ColumnId: columnId,
		}
		ruleColumns = append(ruleColumns, ruleColumn)
	}
	columnId, _ := strconv.ParseInt(data.RhsColumn.ColumnId, 10, 64)
	ruleColumn := po.RuleColumn{
		RuleId:   pgRule.Id,
		BindId:   tableIdi,
		ColumnId: columnId,
	}
	ruleColumns = append(ruleColumns, ruleColumn)
	err = po.CreateRuleColumns(&ruleColumns, db.DB)
	if err != nil {
		fmt.Println(err)
		return 0
	}

	return ruleId
}

// WriteRule2DB 写入规则到数据库中.
// 返回RuleId. 返回0时表示执行出错。
func WriteRule2DB(r rds.Rule, taskId int64, conflictCount, ruleFindType int) int {
	yColumnId := int64(0)
	yColumnId2 := int64(0)
	if r.RuleType != 2 { //正则的没有Y列
		yColumnId, _ = strconv.ParseInt(r.Rhs.LeftColumn.ColumnId, 10, 64)
		yColumnId2, _ = strconv.ParseInt(r.Rhs.RightColumn.ColumnId, 10, 64)
	}
	tidStr := strings.Join(utils.GetRelatedTable(r), ",")
	pgRule := po.Rule{
		Id:            0,
		TaskId:        taskId,
		ConflictCount: conflictCount,
		Ree:           r.Ree,
		ReeJson:       utils.GetRuleJson(r),
		ReeType:       r.RuleType,
		Status:        enum.RULE_UNUSED,
		Support:       r.CR,
		Confidence:    r.FTR,
		YColumnId:     yColumnId,
		YColumnId2:    yColumnId2,
		BindIdList:    tidStr,
		IsUserDefine:  r.IsUserDefine,
	}
	var err error
	if ruleFindType == rds_config.NormalRuleFind {
		err = po.CreateRule(&pgRule, db.DB)
	} else {
		yTableId, _ := strconv.ParseInt(r.Rhs.LeftColumn.TableId, 10, 64)
		err = rule_table.WriteRule2Storage(&pgRule, yTableId)
	}
	if err != nil {
		logger.Error("insert rule error:%v", err)
		return 0
	}
	if ruleFindType == rds_config.NormalRuleFind {
		var ruleColumns []po.RuleColumn
		columns := make(map[string]string)
		for _, predicate := range r.LhsPredicates {
			columns[predicate.LeftColumn.ColumnId] = predicate.LeftColumn.TableId
			columns[predicate.RightColumn.ColumnId] = predicate.RightColumn.TableId
		}
		for columnId, tId := range columns {
			if columnId == "" {
				continue
			}
			columnIdI, _ := strconv.ParseInt(columnId, 10, 64)
			tIdI, _ := strconv.ParseInt(tId, 10, 64)
			ruleColumn := po.RuleColumn{
				RuleId:   pgRule.Id,
				BindId:   tIdI,
				ColumnId: columnIdI,
				LabelY:   0,
				TaskId:   taskId,
			}
			ruleColumns = append(ruleColumns, ruleColumn)
		}
		rhsColumnIds := []string{r.Rhs.LeftColumn.ColumnId, r.Rhs.RightColumn.ColumnId}
		rhsTableId := []string{r.Rhs.LeftColumn.TableId, r.Rhs.RightColumn.TableId}
		for i, columnId := range rhsColumnIds {
			if columnId == "" {
				continue
			}
			if _, ok := columns[columnId]; !ok {
				columnIdI, _ := strconv.ParseInt(columnId, 10, 64)
				tIdI, _ := strconv.ParseInt(rhsTableId[i], 10, 64)
				ruleColumn := po.RuleColumn{
					RuleId:   pgRule.Id,
					BindId:   tIdI,
					ColumnId: columnIdI,
					LabelY:   1,
					TaskId:   taskId,
				}
				ruleColumns = append(ruleColumns, ruleColumn)
				columns[columnId] = rhsTableId[i]
			}
		}
		err = po.CreateRuleColumns(&ruleColumns, db.DB)
		if err != nil {
			logger.Error("insert column error:%v", err)
			return 0
		}

		//for _, column := range r.LhsColumns {
		//	columnId, _ := strconv.ParseInt(column.ColumnId, 10, 64)
		//	labelY := 0
		//	if r.RuleType == 2 {
		//		labelY = 1
		//	}
		//	ruleColumn := po.RuleColumn{
		//		RuleId:   pgRule.Id,
		//		BindId:   tableId,
		//		ColumnId: columnId,
		//		LabelY:   labelY,
		//	}
		//	ruleColumns = append(ruleColumns, ruleColumn)
		//}
		//if r.RuleType != 2 { //正则的没有Y列
		//	//columnId, _ := strconv.ParseInt(r.RhsColumn.ColumnId, 10, 64)
		//	ruleColumn := po.RuleColumn{
		//		RuleId:   pgRule.Id,
		//		BindId:   tableId,
		//		ColumnId: yColumnId,
		//		LabelY:   1,
		//	}
		//	ruleColumns = append(ruleColumns, ruleColumn)
		//}
		//err = po.CreateRuleColumns(&ruleColumns, db.DB)
		//if err != nil {
		//	logger.Error("insert column error:%v", err)
		//	return 0
		//}
	}
	return int(pgRule.Id)
}

func UpdateConflictCount(ruleId int, count int) {
	po.UpdateConflictCount(int64(ruleId), count, db.DB)
}

func UpdateConflictCountTX(ruleId int, count int, tx *gorm.DB) {
	po.UpdateConflictCount(int64(ruleId), count, tx)
}

func GetAbandonedRules() []string {
	return po.GetAbandonedRules()
}

func UpdateConfidence(ruleId int, support, confidence float64, reeJson string) {
	po.UpdateConfidence(int64(ruleId), support, confidence, reeJson, db.DB)
}

func UpdateConfidenceTX(ruleId int, support, confidence float64, reeJson string, tx *gorm.DB) {
	po.UpdateConfidence(int64(ruleId), support, confidence, reeJson, tx)
}

func UpdateRuleStatus(taskId int64, ruleId int64, status int) {
	err := po.UpdateStatus(taskId, ruleId, status, db.DB)
	if err != nil {
		logger.Error("error:%v", err)
		return
	}
}

func UpdateRuleStatusTX(taskId int64, ruleId int64, status int, tx *gorm.DB) {
	err := po.UpdateStatus(taskId, ruleId, status, tx)
	if err != nil {
		logger.Error("error:%v", err)
		return
	}
}
