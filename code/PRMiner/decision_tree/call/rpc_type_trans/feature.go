package rpc_type_trans

import "rds-shenglin/decision_tree/ml/tree"

func FeatureListToThrift(features []tree.FeatureId) []int32 {
	num := len(features)
	to := make([]int32, num)
	for i := 0; i < num; i++ {
		to[i] = int32(features[i])
	}
	return to
}
