
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(animated_box)

# Find packages

find_package(gazebo REQUIRED)

list(APPEND CMAKE_CXX_FLAGS '${GAZEBO_CXX_FLAGS}')

# include appropriate directories
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})

# Create libraries and executables

add_library(R2L_104_0 SHARED R2L_104_0.cc)
target_link_libraries(R2L_104_0 ${GAZEBO_LIBRARIES})

add_library(R2L_107_1 SHARED R2L_107_1.cc)
target_link_libraries(R2L_107_1 ${GAZEBO_LIBRARIES})

add_library(R2L_180_2 SHARED R2L_180_2.cc)
target_link_libraries(R2L_180_2 ${GAZEBO_LIBRARIES})

add_library(R2L_47_3 SHARED R2L_47_3.cc)
target_link_libraries(R2L_47_3 ${GAZEBO_LIBRARIES})

add_library(R2L_56_4 SHARED R2L_56_4.cc)
target_link_libraries(R2L_56_4 ${GAZEBO_LIBRARIES})

add_library(R2L_48_5 SHARED R2L_48_5.cc)
target_link_libraries(R2L_48_5 ${GAZEBO_LIBRARIES})

add_library(R2L_151_6 SHARED R2L_151_6.cc)
target_link_libraries(R2L_151_6 ${GAZEBO_LIBRARIES})

add_library(R2L_81_7 SHARED R2L_81_7.cc)
target_link_libraries(R2L_81_7 ${GAZEBO_LIBRARIES})

add_library(R2L_144_8 SHARED R2L_144_8.cc)
target_link_libraries(R2L_144_8 ${GAZEBO_LIBRARIES})

add_library(R2L_106_9 SHARED R2L_106_9.cc)
target_link_libraries(R2L_106_9 ${GAZEBO_LIBRARIES})

add_library(R2L_144_10 SHARED R2L_144_10.cc)
target_link_libraries(R2L_144_10 ${GAZEBO_LIBRARIES})

add_library(R2L_181_11 SHARED R2L_181_11.cc)
target_link_libraries(R2L_181_11 ${GAZEBO_LIBRARIES})

add_library(R2L_68_12 SHARED R2L_68_12.cc)
target_link_libraries(R2L_68_12 ${GAZEBO_LIBRARIES})

add_library(R2L_88_13 SHARED R2L_88_13.cc)
target_link_libraries(R2L_88_13 ${GAZEBO_LIBRARIES})

add_library(R2L_41_14 SHARED R2L_41_14.cc)
target_link_libraries(R2L_41_14 ${GAZEBO_LIBRARIES})

add_library(R2L_174_15 SHARED R2L_174_15.cc)
target_link_libraries(R2L_174_15 ${GAZEBO_LIBRARIES})

add_library(R2L_128_16 SHARED R2L_128_16.cc)
target_link_libraries(R2L_128_16 ${GAZEBO_LIBRARIES})

add_library(R2L_99_17 SHARED R2L_99_17.cc)
target_link_libraries(R2L_99_17 ${GAZEBO_LIBRARIES})

add_library(R2L_33_18 SHARED R2L_33_18.cc)
target_link_libraries(R2L_33_18 ${GAZEBO_LIBRARIES})

add_library(R2L_142_19 SHARED R2L_142_19.cc)
target_link_libraries(R2L_142_19 ${GAZEBO_LIBRARIES})

add_library(L2R_94_0 SHARED L2R_94_0.cc)
target_link_libraries(L2R_94_0 ${GAZEBO_LIBRARIES})

add_library(L2R_152_1 SHARED L2R_152_1.cc)
target_link_libraries(L2R_152_1 ${GAZEBO_LIBRARIES})

add_library(L2R_132_2 SHARED L2R_132_2.cc)
target_link_libraries(L2R_132_2 ${GAZEBO_LIBRARIES})

add_library(L2R_148_3 SHARED L2R_148_3.cc)
target_link_libraries(L2R_148_3 ${GAZEBO_LIBRARIES})

add_library(L2R_127_4 SHARED L2R_127_4.cc)
target_link_libraries(L2R_127_4 ${GAZEBO_LIBRARIES})

add_library(L2R_42_5 SHARED L2R_42_5.cc)
target_link_libraries(L2R_42_5 ${GAZEBO_LIBRARIES})

add_library(L2R_102_6 SHARED L2R_102_6.cc)
target_link_libraries(L2R_102_6 ${GAZEBO_LIBRARIES})

add_library(L2R_96_7 SHARED L2R_96_7.cc)
target_link_libraries(L2R_96_7 ${GAZEBO_LIBRARIES})

add_library(L2R_157_8 SHARED L2R_157_8.cc)
target_link_libraries(L2R_157_8 ${GAZEBO_LIBRARIES})

add_library(L2R_170_9 SHARED L2R_170_9.cc)
target_link_libraries(L2R_170_9 ${GAZEBO_LIBRARIES})

add_library(L2R_60_10 SHARED L2R_60_10.cc)
target_link_libraries(L2R_60_10 ${GAZEBO_LIBRARIES})

add_library(L2R_100_11 SHARED L2R_100_11.cc)
target_link_libraries(L2R_100_11 ${GAZEBO_LIBRARIES})

add_library(L2R_135_12 SHARED L2R_135_12.cc)
target_link_libraries(L2R_135_12 ${GAZEBO_LIBRARIES})

add_library(L2R_64_13 SHARED L2R_64_13.cc)
target_link_libraries(L2R_64_13 ${GAZEBO_LIBRARIES})

add_library(L2R_100_14 SHARED L2R_100_14.cc)
target_link_libraries(L2R_100_14 ${GAZEBO_LIBRARIES})

add_library(L2R_43_15 SHARED L2R_43_15.cc)
target_link_libraries(L2R_43_15 ${GAZEBO_LIBRARIES})

add_library(L2R_185_16 SHARED L2R_185_16.cc)
target_link_libraries(L2R_185_16 ${GAZEBO_LIBRARIES})

add_library(L2R_135_17 SHARED L2R_135_17.cc)
target_link_libraries(L2R_135_17 ${GAZEBO_LIBRARIES})

add_library(L2R_115_18 SHARED L2R_115_18.cc)
target_link_libraries(L2R_115_18 ${GAZEBO_LIBRARIES})

add_library(L2R_75_19 SHARED L2R_75_19.cc)
target_link_libraries(L2R_75_19 ${GAZEBO_LIBRARIES})

add_library(T2B_145_0 SHARED T2B_145_0.cc)
target_link_libraries(T2B_145_0 ${GAZEBO_LIBRARIES})

add_library(T2B_133_1 SHARED T2B_133_1.cc)
target_link_libraries(T2B_133_1 ${GAZEBO_LIBRARIES})

add_library(T2B_185_2 SHARED T2B_185_2.cc)
target_link_libraries(T2B_185_2 ${GAZEBO_LIBRARIES})

add_library(T2B_50_3 SHARED T2B_50_3.cc)
target_link_libraries(T2B_50_3 ${GAZEBO_LIBRARIES})

add_library(T2B_175_4 SHARED T2B_175_4.cc)
target_link_libraries(T2B_175_4 ${GAZEBO_LIBRARIES})

add_library(T2B_68_5 SHARED T2B_68_5.cc)
target_link_libraries(T2B_68_5 ${GAZEBO_LIBRARIES})

add_library(T2B_125_6 SHARED T2B_125_6.cc)
target_link_libraries(T2B_125_6 ${GAZEBO_LIBRARIES})

add_library(T2B_66_7 SHARED T2B_66_7.cc)
target_link_libraries(T2B_66_7 ${GAZEBO_LIBRARIES})

add_library(T2B_110_8 SHARED T2B_110_8.cc)
target_link_libraries(T2B_110_8 ${GAZEBO_LIBRARIES})

add_library(T2B_178_9 SHARED T2B_178_9.cc)
target_link_libraries(T2B_178_9 ${GAZEBO_LIBRARIES})

add_library(T2B_87_10 SHARED T2B_87_10.cc)
target_link_libraries(T2B_87_10 ${GAZEBO_LIBRARIES})

add_library(T2B_147_11 SHARED T2B_147_11.cc)
target_link_libraries(T2B_147_11 ${GAZEBO_LIBRARIES})

add_library(T2B_67_12 SHARED T2B_67_12.cc)
target_link_libraries(T2B_67_12 ${GAZEBO_LIBRARIES})

add_library(T2B_165_13 SHARED T2B_165_13.cc)
target_link_libraries(T2B_165_13 ${GAZEBO_LIBRARIES})

add_library(T2B_38_14 SHARED T2B_38_14.cc)
target_link_libraries(T2B_38_14 ${GAZEBO_LIBRARIES})

add_library(T2B_102_15 SHARED T2B_102_15.cc)
target_link_libraries(T2B_102_15 ${GAZEBO_LIBRARIES})

add_library(T2B_86_16 SHARED T2B_86_16.cc)
target_link_libraries(T2B_86_16 ${GAZEBO_LIBRARIES})

add_library(T2B_80_17 SHARED T2B_80_17.cc)
target_link_libraries(T2B_80_17 ${GAZEBO_LIBRARIES})

add_library(T2B_87_18 SHARED T2B_87_18.cc)
target_link_libraries(T2B_87_18 ${GAZEBO_LIBRARIES})

add_library(T2B_163_19 SHARED T2B_163_19.cc)
target_link_libraries(T2B_163_19 ${GAZEBO_LIBRARIES})

add_library(B2T_142_0 SHARED B2T_142_0.cc)
target_link_libraries(B2T_142_0 ${GAZEBO_LIBRARIES})

add_library(B2T_60_1 SHARED B2T_60_1.cc)
target_link_libraries(B2T_60_1 ${GAZEBO_LIBRARIES})

add_library(B2T_76_2 SHARED B2T_76_2.cc)
target_link_libraries(B2T_76_2 ${GAZEBO_LIBRARIES})

add_library(B2T_77_3 SHARED B2T_77_3.cc)
target_link_libraries(B2T_77_3 ${GAZEBO_LIBRARIES})

add_library(B2T_150_4 SHARED B2T_150_4.cc)
target_link_libraries(B2T_150_4 ${GAZEBO_LIBRARIES})

add_library(B2T_77_5 SHARED B2T_77_5.cc)
target_link_libraries(B2T_77_5 ${GAZEBO_LIBRARIES})

add_library(B2T_40_6 SHARED B2T_40_6.cc)
target_link_libraries(B2T_40_6 ${GAZEBO_LIBRARIES})

add_library(B2T_120_7 SHARED B2T_120_7.cc)
target_link_libraries(B2T_120_7 ${GAZEBO_LIBRARIES})

add_library(B2T_174_8 SHARED B2T_174_8.cc)
target_link_libraries(B2T_174_8 ${GAZEBO_LIBRARIES})

add_library(B2T_86_9 SHARED B2T_86_9.cc)
target_link_libraries(B2T_86_9 ${GAZEBO_LIBRARIES})

add_library(B2T_156_10 SHARED B2T_156_10.cc)
target_link_libraries(B2T_156_10 ${GAZEBO_LIBRARIES})

add_library(B2T_174_11 SHARED B2T_174_11.cc)
target_link_libraries(B2T_174_11 ${GAZEBO_LIBRARIES})

add_library(B2T_129_12 SHARED B2T_129_12.cc)
target_link_libraries(B2T_129_12 ${GAZEBO_LIBRARIES})

add_library(B2T_65_13 SHARED B2T_65_13.cc)
target_link_libraries(B2T_65_13 ${GAZEBO_LIBRARIES})

add_library(B2T_108_14 SHARED B2T_108_14.cc)
target_link_libraries(B2T_108_14 ${GAZEBO_LIBRARIES})

add_library(B2T_136_15 SHARED B2T_136_15.cc)
target_link_libraries(B2T_136_15 ${GAZEBO_LIBRARIES})

add_library(B2T_96_16 SHARED B2T_96_16.cc)
target_link_libraries(B2T_96_16 ${GAZEBO_LIBRARIES})

add_library(B2T_33_17 SHARED B2T_33_17.cc)
target_link_libraries(B2T_33_17 ${GAZEBO_LIBRARIES})

add_library(B2T_171_18 SHARED B2T_171_18.cc)
target_link_libraries(B2T_171_18 ${GAZEBO_LIBRARIES})

add_library(B2T_99_19 SHARED B2T_99_19.cc)
target_link_libraries(B2T_99_19 ${GAZEBO_LIBRARIES})
