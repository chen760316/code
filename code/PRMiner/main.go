package main

import (
	"fmt"
	"github.com/gin-gonic/gin"
	"net/http"
	"rds-shenglin/rds_config"
	"rds-shenglin/rock-share/base/config"
	"rds-shenglin/rock-share/base/logger"
)

func main() {
	go func() {
		err := http.ListenAndServe(":8081", nil)
		if err != nil {
			fmt.Printf("http.ListenAndServe failed, err:%s", err)
		}
	}()

	// 一些初始化配置
	config.InitConfig()
	all := config.All
	l := all.Logger
	ss := all.Server
	logger.InitLogger(l.Level, "rock", l.Path, l.MaxAge, l.RotationTime, l.RotationSize, ss.SentryDsn)
	r := gin.Default()

	r.POST("/rds", start)

	address := ":" + rds_config.GinPort
	r.Run(address)
}

func start(c *gin.Context) {
	var requestJson RDSRequest
	if err := c.ShouldBindJSON(&requestJson); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": err.Error(),
		})
		fmt.Println("_____________________请求异常:")
		fmt.Println(err)
		return
	}
	p, size, t, e := DigRule(&requestJson)
	if e != nil {
		c.JSON(http.StatusOK, gin.H{
			"success": false,
			"error":   e,
		})
	} else {
		c.JSON(http.StatusOK, gin.H{
			"success":     true,
			"result_path": p,
			"rule_size":   size,
			"spent_time":  t,
		})
	}

}
