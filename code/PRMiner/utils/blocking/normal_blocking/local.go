package normal_blocking

import (
	"rds-shenglin/rock-share/base/logger"
	"rds-shenglin/utils/blocking/blocking_conf"
	"rds-shenglin/storage/storage2/database/sql"
	"rds-shenglin/storage/storage2/rock_db"
	"rds-shenglin/storage/storage2/utils"
	"time"
)

func LocalMemoryBlockingCrossColumn(taskId int64, blockIndex int, leftTableName, leftColumnName, rightTableName, rightColumnName string) error {
	{
		filePath := blocking_conf.FilePath(taskId, blockIndex, leftTableName, leftColumnName, 0, blocking_conf.NormalBlockingFileSuffix)
		exist, err := rock_db.DFS.Exist(filePath)
		if err != nil {
			logger.Error(err)
			return err
		}
		if exist {
			logger.Info("blocking ", leftTableName, "-", leftColumnName, " ", rightTableName, "-", rightColumnName, " 已存在，跳过")
			return nil
		}
	}

	{
		logger.Info("开始 blocking ", leftTableName, "-", leftColumnName, " ", rightTableName, "-", rightColumnName)
		go func(time time.Time) {
			logger.Info("开始 blocking ", leftTableName, "-", leftColumnName, " ", rightTableName, "-", rightColumnName, " 完成，耗时", utils.DurationString(time))
		}(time.Now())
	}
	var leftTableLength int64 = 0
	var rightTableLength int64 = 0
	{
		lenW, err := rock_db.DB.Query(sql.SingleTable.SELECT(sql.CountAsterisk).FROM(leftTableName))
		if err != nil {
			logger.Error(err)
			return err
		}
		leftTableLength = lenW[0][0].(int64)
	}
	{
		lenW, err := rock_db.DB.Query(sql.SingleTable.SELECT(sql.CountAsterisk).FROM(rightTableName))
		if err != nil {
			logger.Error(err)
			return err
		}
		rightTableLength = lenW[0][0].(int64)
	}

	var tokenCntMap = map[Token]Frequency{}
	for batchId := int64(0); true; batchId++ {
		start := batchId * blocking_conf.FileRowSize
		end := start + blocking_conf.FileRowSize
		if start >= leftTableLength {
			break
		}
		if end > leftTableLength {
			end = leftTableLength
		}
		logger.Info("表 ", leftTableName, " 列 ", leftColumnName, " 范围 ", start, " ~ ", end, "的数据计算 token")
		batchTokenCntMap, err := Tokenize(leftTableName, leftColumnName, start, end-start) // 内部并行
		if err != nil {
			logger.Error(err)
			return err
		}
		TokenFreqCombine(tokenCntMap, batchTokenCntMap)
		logger.Info("blocking 合并，当前 token 数目 ", len(tokenCntMap))

		if end == leftTableLength {
			break
		}
	}
	for batchId := int64(0); true; batchId++ {
		start := batchId * blocking_conf.FileRowSize
		end := start + blocking_conf.FileRowSize
		if start >= rightTableLength {
			break
		}
		if end > rightTableLength {
			end = rightTableLength
		}
		logger.Info("表 ", rightTableName, " 列 ", rightColumnName, " 范围 ", start, " ~ ", end, "的数据计算 token")
		batchTokenCntMap, err := Tokenize(rightTableName, rightColumnName, start, end-start) // 内部并行
		if err != nil {
			logger.Error(err)
			return err
		}
		TokenFreqCombine(tokenCntMap, batchTokenCntMap)
		logger.Info("blocking 合并，当前 token 数目 ", len(tokenCntMap))

		if end == rightTableLength {
			break
		}
	}
	logger.Info("token 计算完成，开始排序和编号")
	//for token, frequency := range tokenCntMap {
	//	if int64(frequency) > (leftTableLength+rightTableLength)/2 {
	//		tokenCntMap[token] = 1
	//	}
	//}
	sortedTokens := TokenFreqSort(tokenCntMap)
	logger.Info("token 排序完成，去除了只出现一次的 token，数目", len(sortedTokens))

	for shardId := int64(0); true; shardId++ {
		startRowId := shardId * blocking_conf.FileRowSize
		if startRowId >= leftTableLength {
			break
		}
		logger.Info("生成tokenId列，当前范围 ", startRowId, "~", startRowId+blocking_conf.FileRowSize)
		_, err := CreateDerivedColumn(taskId, blockIndex, leftTableName, leftColumnName, startRowId, sortedTokens)
		if err != nil {
			logger.Error(err)
			return err
		}
	}
	for shardId := int64(0); true; shardId++ {
		startRowId := shardId * blocking_conf.FileRowSize
		if startRowId >= rightTableLength {
			break
		}
		logger.Info("生成tokenId列，当前范围 ", startRowId, "~", startRowId+blocking_conf.FileRowSize)
		_, err := CreateDerivedColumn(taskId, blockIndex, rightTableName, rightColumnName, startRowId, sortedTokens)
		if err != nil {
			logger.Error(err)
			return err
		}
	}
	return nil
}

func LocalMemoryBlockingCrossColumnNew(taskId int64, blockIndex int, leftTableName, leftColumnName, rightTableName, rightColumnName string, tableColumnValues map[string]map[string][]interface{}) error {
	{
		filePath := blocking_conf.FilePath(taskId, blockIndex, leftTableName, leftColumnName, 0, blocking_conf.NormalBlockingFileSuffix)
		_, err := rock_db.DFS.Exist(filePath)
		if err != nil {
			logger.Error(err)
			return err
		}
		//if exist {
		//	logger.Info("blocking ", leftTableName, "-", leftColumnName, " ", rightTableName, "-", rightColumnName, " 已存在，跳过")
		//	return nil
		//}
	}

	{
		logger.Info("开始 blocking ", leftTableName, "-", leftColumnName, " ", rightTableName, "-", rightColumnName)
		go func(time time.Time) {
			logger.Info("开始 blocking ", leftTableName, "-", leftColumnName, " ", rightTableName, "-", rightColumnName, " 完成，耗时", utils.DurationString(time))
		}(time.Now())
	}
	var leftTableLength int64 = 0
	var rightTableLength int64 = 0
	{
		for _, values := range tableColumnValues[leftTableName] {
			leftTableLength = int64(len(values))
			break
		}
	}
	{

		for _, values := range tableColumnValues[rightTableName] {
			rightTableLength = int64(len(values))
			break
		}
	}

	var tokenCntMap = map[Token]Frequency{}
	for batchId := int64(0); true; batchId++ {
		start := batchId * blocking_conf.FileRowSize
		end := start + blocking_conf.FileRowSize
		if start >= leftTableLength {
			break
		}
		if end > leftTableLength {
			end = leftTableLength
		}
		logger.Info("表 ", leftTableName, " 列 ", leftColumnName, " 范围 ", start, " ~ ", end, "的数据计算 token")
		batchTokenCntMap, err := TokenizeNew(tableColumnValues[leftTableName][leftColumnName], start, end-start) // 内部并行
		if err != nil {
			logger.Error(err)
			return err
		}
		TokenFreqCombine(tokenCntMap, batchTokenCntMap)
		logger.Info("blocking 合并，当前 token 数目 ", len(tokenCntMap))

		if end == leftTableLength {
			break
		}
	}
	for batchId := int64(0); true; batchId++ {
		start := batchId * blocking_conf.FileRowSize
		end := start + blocking_conf.FileRowSize
		if start >= rightTableLength {
			break
		}
		if end > rightTableLength {
			end = rightTableLength
		}
		logger.Info("表 ", rightTableName, " 列 ", rightColumnName, " 范围 ", start, " ~ ", end, "的数据计算 token")
		batchTokenCntMap, err := TokenizeNew(tableColumnValues[rightTableName][rightColumnName], start, end-start) // 内部并行
		if err != nil {
			logger.Error(err)
			return err
		}
		TokenFreqCombine(tokenCntMap, batchTokenCntMap)
		logger.Info("blocking 合并，当前 token 数目 ", len(tokenCntMap))

		if end == rightTableLength {
			break
		}
	}
	logger.Info("token 计算完成，开始排序和编号")
	//for token, frequency := range tokenCntMap {
	//	if int64(frequency) > (leftTableLength+rightTableLength)/2 {
	//		tokenCntMap[token] = 1
	//	}
	//}
	sortedTokens := TokenFreqSort(tokenCntMap)
	logger.Info("token 排序完成，去除了只出现一次的 token，数目", len(sortedTokens))

	for shardId := int64(0); true; shardId++ {
		startRowId := shardId * blocking_conf.FileRowSize
		if startRowId >= leftTableLength {
			break
		}
		logger.Info("生成tokenId列，当前范围 ", startRowId, "~", startRowId+blocking_conf.FileRowSize)
		_, err := CreateDerivedColumnNew(taskId, blockIndex, leftTableName, leftColumnName, tableColumnValues[leftTableName][leftColumnName], startRowId, sortedTokens)
		if err != nil {
			logger.Error(err)
			return err
		}
	}
	for shardId := int64(0); true; shardId++ {
		startRowId := shardId * blocking_conf.FileRowSize
		if startRowId >= rightTableLength {
			break
		}
		logger.Info("生成tokenId列，当前范围 ", startRowId, "~", startRowId+blocking_conf.FileRowSize)
		_, err := CreateDerivedColumnNew(taskId, blockIndex, rightTableName, rightColumnName, tableColumnValues[rightTableName][rightColumnName], startRowId, sortedTokens)
		if err != nil {
			logger.Error(err)
			return err
		}
	}
	return nil
}
