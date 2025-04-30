package main

type RDSRequest struct {
	Table      Table   `json:"table"`
	Support    float64 `json:"support"`
	Confidence float64 `json:"confidence"`
}

type Table struct {
	Path        string            `json:"path"`
	ColumnsType map[string]string `json:"columnsType"`
}
