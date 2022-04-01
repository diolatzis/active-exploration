## ======================================================================== ##
## Copyright 2009-2020 Intel Corporation                                    ##
##                                                                          ##
## Licensed under the Apache License, Version 2.0 (the "License");          ##
## you may not use this file except in compliance with the License.         ##
## You may obtain a copy of the License at                                  ##
##                                                                          ##
##     http://www.apache.org/licenses/LICENSE-2.0                           ##
##                                                                          ##
## Unless required by applicable law or agreed to in writing, software      ##
## distributed under the License is distributed on an "AS IS" BASIS,        ##
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. ##
## See the License for the specific language governing permissions and      ##
## limitations under the License.                                           ##
## ======================================================================== ##

IF (WIN32 OR APPLE)
   return()
ENDIF()

execute_process(COMMAND objdump -C -t ${file} OUTPUT_VARIABLE output)
string(REPLACE "\n" ";" output ${output})

foreach (line ${output})
  if ("${line}" MATCHES "O .bss")
    if (NOT "${line}" MATCHES "std::__ioinit" AND          # this is caused by iostream initialization and is likely also ok
        NOT "${line}" MATCHES "\\(\\)::" AND               # this matches a static inside a function which is fine
        NOT "${line}" MATCHES "function_local_static_" AND # static variable inside a function (explicitely named)
        NOT "${line}" MATCHES "__\\$U")                    # ICC generated locks for static variable inside a function
      message(WARNING "\nProblematic global variable in non-SSE code:\n" ${line})
    endif()
  endif()
endforeach()
