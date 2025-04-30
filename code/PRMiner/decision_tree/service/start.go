package main

import (
	"rds-shenglin/rock-share/base/logger"
	"rds-shenglin/decision_tree/call"
	"rds-shenglin/decision_tree/conf/tree"
	"rds-shenglin/decision_tree/param/conf_cluster"
	"net"
)

func main() {
	//conf.Init()

	//conf_cluster.ClusterConfigInit(os.Args[1:])

	//common.InitRpcParam(
	//	conf_cluster.Cluster.RpcProtocol,
	//	conf_cluster.Cluster.RpcFramed,
	//	conf_cluster.Cluster.RpcBuffered,
	//	conf_cluster.Cluster.RpcBufferSize,
	//	conf_cluster.Cluster.RpcSecure,
	//)

	//log.Info().Msg(version.VERSION)
	//log.Info().Msg(conf_cluster.ClusterSettingsToString())

	//serviceBeginTime := time.Now()

	//InitDistributedSystem()

	// 启动水平扩展worker服务
	_, _, err := call.StartWorkerServer(
		"xxx",
		net.JoinHostPort(conf_cluster.Cluster.LocalIp, conf_cluster.Cluster.HspawnWorkerServicePort),
		[]string{conf_cluster.Cluster.EtcdAddr},
		tree.ImpurityCriterion,
	)
	if err != nil {
		logger.Error(err.Error())
		return
	}
	//log.Info().Msg("service start!!!")

	//defer func(startTime time.Time) {
	//	log.Info().Msgf("serve time: %v", time.Since(startTime))
	//}(serviceBeginTime)

}
