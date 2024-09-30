#!/usr/bin/env bash

joern=~/bin/joern/joern-cli/joern
joern_cpg2scpg=~/bin/joern/joern-cli/joern-cpg2scpg
joern_export=~/bin/joern/joern-cli/joern-export
joern_flow=~/bin/joern/joern-cli/joern-flow
joern_parse=~/bin/joern/joern-cli/joern-parse
joern_scan=~/bin/joern/joern-cli/joern-scan
joern_slice=~/bin/joern/joern-cli/joern-slice
joern_vectors=~/bin/joern/joern-cli/joern-vectors

rm -rf ./workspace/

${joern} -J-Xmx10G --script extract_implementation.sc > /dev/null 2>&1

printf 'Done!\n'

bat call_stack_trace.txt
