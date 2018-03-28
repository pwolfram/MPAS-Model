#!/usr/bin/env bash

rmtrash tmp_wettingdrying/
./manage_regression_suite.py -f general.config.ocean.pn1704857 --work_dir=${PWD}/tmp_wettingdrying -c -s -t ocean/regression_suites/wettingdrying.xml
cd tmp_wettingdrying/
./nightly_ocean_test_suite.py
cd ..
