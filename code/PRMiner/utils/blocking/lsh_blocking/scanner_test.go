package lsh_blocking

import (
	"bufio"
	"strings"
	"testing"
)

func TestScanner(t *testing.T) {
	var s = "apple\r\norange\r\npear"
	reader := strings.NewReader(s)
	scanner := bufio.NewScanner(reader)
	for scanner.Scan() {
		text := scanner.Text()
		println(text)
	}
}
