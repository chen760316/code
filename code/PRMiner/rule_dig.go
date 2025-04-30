package main

import (
	"path"
	"rds-shenglin/rds_config"
	"rds-shenglin/rock-share/base/logger"
	"rds-shenglin/utils"
	"strconv"
	"strings"
	"sync"
	"time"
)

func DigRule(request *RDSRequest) (string, int, int64, error) {
	startTime := time.Now().UnixMilli()
	taskId := startTime
	logger.Infof("规则发现开始")

	gv := InitRdsGlobalV(taskId, request.Table, request.Support, request.Confidence)
	gv.MutexGroup = make(map[string][]int)

	// 生成表内的结构性谓词和决策树需要跑的Y列
	GenerateInnerPredicateNew(gv)

	if len(gv.Predicates) == 0 && len(gv.CrossTablePredicates[1]) == 0 {
		logger.Infof("taskid:%v,生成的谓词数量为0,不需要进行规则发现", taskId)
		ClearMemory(taskId)
		return "", 0, 0, nil
	}

	// 可以作为Y出现的谓词
	root := gv.Predicates

	logger.Infof("task id:%v, 普通谓词:%v", taskId, utils.GetPredicatesStr(gv.Predicates))
	logger.Infof("task id:%v, 主外键谓词:%v", taskId, utils.GetPredicatesStr(gv.CrossTablePredicates[0]))
	logger.Infof("task id:%v, 其他跨列谓词谓词:%v", taskId, utils.GetPredicatesStr(gv.CrossTablePredicates[1]))

	// 单行规则发现
	if gv.enableDecisionTree {
		t := time.Now().UnixMilli()
		logger.Infof("task id:%v, 单行规则发现开始", taskId)
		singleRowRuleDig(gv)
		if gv.HasError {
			ClearMemory(taskId)
			return "", 0, 0, nil
		}
		singleRowRuleTime := time.Now().UnixMilli() - t
		logger.Infof("taskid:%v,单行规则执行时间:%vms, 发现的规则数:%v", gv.TaskId, singleRowRuleTime, gv.SingleRuleSize)
	}

	BuildTrees(root, gv)

	// 过滤confidence为1和用户自定义的规则
	p := ExeRules(gv)

	logger.Infof("taskId:%v规则发现已完成,耗时%dms, 单行规则:%v;多行规则:%v", taskId, time.Now().UnixMilli()-startTime, gv.SingleRuleSize, gv.MultiRuleSize)
	return p, len(gv.rules), time.Now().UnixMilli() - startTime, nil
}

func singleRowRuleDig(gv *GlobalV) {
	//0.生成备选Y，已生成
	//1.分发任务
	wg := sync.WaitGroup{}
	for tableId, yColumns := range gv.Table2DecisionY {
		rowSize := gv.RowSizes[tableId]
		columnsType := gv.ColumnsType[tableId]
		for _, yColumn := range yColumns {
			if (columnsType[yColumn] == rds_config.IntType || columnsType[yColumn] == rds_config.FloatType) && !gv.enableNum {
				continue
			}
			if (columnsType[yColumn] == rds_config.EnumType || columnsType[yColumn] == rds_config.BoolType) && !gv.enableEnum {
				continue
			}
			wg.Add(1)
			go func(tableId, yCol string, column2Type map[string]string, columnLength int) {
				defer wg.Done()
				ruleSize := ExecuteSingleRowDecisionTree(tableId, yCol, column2Type, columnLength, gv)
				gv.RuleSizeLock.Lock()
				gv.SingleRuleSize += ruleSize
				gv.RuleSizeLock.Unlock()
				logger.Infof("taskId: %v, tableId: %s, yColumn: %s, finish decision tre, ruleSize: %v", gv.TaskId, tableId, yCol, ruleSize)
			}(tableId, yColumn, columnsType, rowSize)
		}
	}
	wg.Wait()
}

func ExeRules(gv *GlobalV) string {
	var data [][]string
	data = append(data, []string{"ree", "support", "confidence", "error"})
	for _, rule := range gv.rules {
		var row []string
		row = append(row, rule.Ree)
		row = append(row, strconv.FormatFloat(rule.CR, 'f', -1, 64))
		row = append(row, strconv.FormatFloat(rule.FTR, 'f', -1, 64))
		// 现在confidence为1的规则不做处理
		if rule.FTR >= 1 {
			row = append(row, "")
			continue
		} else {
			_, _, conflictDataAgg, _ := ExecuteCheckErrorRds(rule, gv)
			var arr []string
			for _, value := range conflictDataAgg {
				for index := range value {
					id := strconv.FormatInt(int64(index), 10)
					arr = append(arr, id)
				}
			}
			row = append(row, strings.Join(arr, ","))
		}
		data = append(data, row)
	}
	p := path.Join("result", strconv.FormatInt(gv.TaskId, 10)+".csv")
	utils.CreateCsv(p, data)
	return p
}
