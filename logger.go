package main

import (
	"go.uber.org/zap"
)

var logger, err = zap.NewDevelopment()

var lg = logger.Sugar()
