
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(animated_box)

# Find packages

find_package(gazebo REQUIRED)

list(APPEND CMAKE_CXX_FLAGS '${GAZEBO_CXX_FLAGS}')

# include appropriate directories
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})

# Create libraries and executables

add_library(R2L_51_0 SHARED R2L_51_0.cc)
target_link_libraries(R2L_51_0 ${GAZEBO_LIBRARIES})

add_library(R2L_53_1 SHARED R2L_53_1.cc)
target_link_libraries(R2L_53_1 ${GAZEBO_LIBRARIES})

add_library(R2L_94_2 SHARED R2L_94_2.cc)
target_link_libraries(R2L_94_2 ${GAZEBO_LIBRARIES})

add_library(R2L_19_3 SHARED R2L_19_3.cc)
target_link_libraries(R2L_19_3 ${GAZEBO_LIBRARIES})

add_library(R2L_24_4 SHARED R2L_24_4.cc)
target_link_libraries(R2L_24_4 ${GAZEBO_LIBRARIES})

add_library(R2L_20_5 SHARED R2L_20_5.cc)
target_link_libraries(R2L_20_5 ${GAZEBO_LIBRARIES})

add_library(R2L_78_6 SHARED R2L_78_6.cc)
target_link_libraries(R2L_78_6 ${GAZEBO_LIBRARIES})

add_library(R2L_38_7 SHARED R2L_38_7.cc)
target_link_libraries(R2L_38_7 ${GAZEBO_LIBRARIES})

add_library(R2L_74_8 SHARED R2L_74_8.cc)
target_link_libraries(R2L_74_8 ${GAZEBO_LIBRARIES})

add_library(R2L_53_9 SHARED R2L_53_9.cc)
target_link_libraries(R2L_53_9 ${GAZEBO_LIBRARIES})

add_library(R2L_74_10 SHARED R2L_74_10.cc)
target_link_libraries(R2L_74_10 ${GAZEBO_LIBRARIES})

add_library(R2L_95_11 SHARED R2L_95_11.cc)
target_link_libraries(R2L_95_11 ${GAZEBO_LIBRARIES})

add_library(R2L_31_12 SHARED R2L_31_12.cc)
target_link_libraries(R2L_31_12 ${GAZEBO_LIBRARIES})

add_library(R2L_42_13 SHARED R2L_42_13.cc)
target_link_libraries(R2L_42_13 ${GAZEBO_LIBRARIES})

add_library(R2L_16_14 SHARED R2L_16_14.cc)
target_link_libraries(R2L_16_14 ${GAZEBO_LIBRARIES})

add_library(R2L_91_15 SHARED R2L_91_15.cc)
target_link_libraries(R2L_91_15 ${GAZEBO_LIBRARIES})

add_library(R2L_65_16 SHARED R2L_65_16.cc)
target_link_libraries(R2L_65_16 ${GAZEBO_LIBRARIES})

add_library(R2L_49_17 SHARED R2L_49_17.cc)
target_link_libraries(R2L_49_17 ${GAZEBO_LIBRARIES})

add_library(R2L_11_18 SHARED R2L_11_18.cc)
target_link_libraries(R2L_11_18 ${GAZEBO_LIBRARIES})

add_library(R2L_73_19 SHARED R2L_73_19.cc)
target_link_libraries(R2L_73_19 ${GAZEBO_LIBRARIES})

add_library(L2R_46_0 SHARED L2R_46_0.cc)
target_link_libraries(L2R_46_0 ${GAZEBO_LIBRARIES})

add_library(L2R_78_1 SHARED L2R_78_1.cc)
target_link_libraries(L2R_78_1 ${GAZEBO_LIBRARIES})

add_library(L2R_67_2 SHARED L2R_67_2.cc)
target_link_libraries(L2R_67_2 ${GAZEBO_LIBRARIES})

add_library(L2R_76_3 SHARED L2R_76_3.cc)
target_link_libraries(L2R_76_3 ${GAZEBO_LIBRARIES})

add_library(L2R_65_4 SHARED L2R_65_4.cc)
target_link_libraries(L2R_65_4 ${GAZEBO_LIBRARIES})

add_library(L2R_16_5 SHARED L2R_16_5.cc)
target_link_libraries(L2R_16_5 ${GAZEBO_LIBRARIES})

add_library(L2R_50_6 SHARED L2R_50_6.cc)
target_link_libraries(L2R_50_6 ${GAZEBO_LIBRARIES})

add_library(L2R_47_7 SHARED L2R_47_7.cc)
target_link_libraries(L2R_47_7 ${GAZEBO_LIBRARIES})

add_library(L2R_81_8 SHARED L2R_81_8.cc)
target_link_libraries(L2R_81_8 ${GAZEBO_LIBRARIES})

add_library(L2R_88_9 SHARED L2R_88_9.cc)
target_link_libraries(L2R_88_9 ${GAZEBO_LIBRARIES})

add_library(L2R_27_10 SHARED L2R_27_10.cc)
target_link_libraries(L2R_27_10 ${GAZEBO_LIBRARIES})

add_library(L2R_49_11 SHARED L2R_49_11.cc)
target_link_libraries(L2R_49_11 ${GAZEBO_LIBRARIES})

add_library(L2R_69_12 SHARED L2R_69_12.cc)
target_link_libraries(L2R_69_12 ${GAZEBO_LIBRARIES})

add_library(L2R_29_13 SHARED L2R_29_13.cc)
target_link_libraries(L2R_29_13 ${GAZEBO_LIBRARIES})

add_library(L2R_49_14 SHARED L2R_49_14.cc)
target_link_libraries(L2R_49_14 ${GAZEBO_LIBRARIES})

add_library(L2R_17_15 SHARED L2R_17_15.cc)
target_link_libraries(L2R_17_15 ${GAZEBO_LIBRARIES})

add_library(L2R_97_16 SHARED L2R_97_16.cc)
target_link_libraries(L2R_97_16 ${GAZEBO_LIBRARIES})

add_library(L2R_69_17 SHARED L2R_69_17.cc)
target_link_libraries(L2R_69_17 ${GAZEBO_LIBRARIES})

add_library(L2R_58_18 SHARED L2R_58_18.cc)
target_link_libraries(L2R_58_18 ${GAZEBO_LIBRARIES})

add_library(L2R_35_19 SHARED L2R_35_19.cc)
target_link_libraries(L2R_35_19 ${GAZEBO_LIBRARIES})

add_library(T2B_75_0 SHARED T2B_75_0.cc)
target_link_libraries(T2B_75_0 ${GAZEBO_LIBRARIES})

add_library(T2B_68_1 SHARED T2B_68_1.cc)
target_link_libraries(T2B_68_1 ${GAZEBO_LIBRARIES})

add_library(T2B_97_2 SHARED T2B_97_2.cc)
target_link_libraries(T2B_97_2 ${GAZEBO_LIBRARIES})

add_library(T2B_21_3 SHARED T2B_21_3.cc)
target_link_libraries(T2B_21_3 ${GAZEBO_LIBRARIES})

add_library(T2B_92_4 SHARED T2B_92_4.cc)
target_link_libraries(T2B_92_4 ${GAZEBO_LIBRARIES})

add_library(T2B_31_5 SHARED T2B_31_5.cc)
target_link_libraries(T2B_31_5 ${GAZEBO_LIBRARIES})

add_library(T2B_63_6 SHARED T2B_63_6.cc)
target_link_libraries(T2B_63_6 ${GAZEBO_LIBRARIES})

add_library(T2B_30_7 SHARED T2B_30_7.cc)
target_link_libraries(T2B_30_7 ${GAZEBO_LIBRARIES})

add_library(T2B_55_8 SHARED T2B_55_8.cc)
target_link_libraries(T2B_55_8 ${GAZEBO_LIBRARIES})

add_library(T2B_93_9 SHARED T2B_93_9.cc)
target_link_libraries(T2B_93_9 ${GAZEBO_LIBRARIES})

add_library(T2B_42_10 SHARED T2B_42_10.cc)
target_link_libraries(T2B_42_10 ${GAZEBO_LIBRARIES})

add_library(T2B_76_11 SHARED T2B_76_11.cc)
target_link_libraries(T2B_76_11 ${GAZEBO_LIBRARIES})

add_library(T2B_30_12 SHARED T2B_30_12.cc)
target_link_libraries(T2B_30_12 ${GAZEBO_LIBRARIES})

add_library(T2B_86_13 SHARED T2B_86_13.cc)
target_link_libraries(T2B_86_13 ${GAZEBO_LIBRARIES})

add_library(T2B_14_14 SHARED T2B_14_14.cc)
target_link_libraries(T2B_14_14 ${GAZEBO_LIBRARIES})

add_library(T2B_50_15 SHARED T2B_50_15.cc)
target_link_libraries(T2B_50_15 ${GAZEBO_LIBRARIES})

add_library(T2B_42_16 SHARED T2B_42_16.cc)
target_link_libraries(T2B_42_16 ${GAZEBO_LIBRARIES})

add_library(T2B_38_17 SHARED T2B_38_17.cc)
target_link_libraries(T2B_38_17 ${GAZEBO_LIBRARIES})

add_library(T2B_42_18 SHARED T2B_42_18.cc)
target_link_libraries(T2B_42_18 ${GAZEBO_LIBRARIES})

add_library(T2B_84_19 SHARED T2B_84_19.cc)
target_link_libraries(T2B_84_19 ${GAZEBO_LIBRARIES})

add_library(B2T_73_0 SHARED B2T_73_0.cc)
target_link_libraries(B2T_73_0 ${GAZEBO_LIBRARIES})

add_library(B2T_27_1 SHARED B2T_27_1.cc)
target_link_libraries(B2T_27_1 ${GAZEBO_LIBRARIES})

add_library(B2T_36_2 SHARED B2T_36_2.cc)
target_link_libraries(B2T_36_2 ${GAZEBO_LIBRARIES})

add_library(B2T_36_3 SHARED B2T_36_3.cc)
target_link_libraries(B2T_36_3 ${GAZEBO_LIBRARIES})

add_library(B2T_77_4 SHARED B2T_77_4.cc)
target_link_libraries(B2T_77_4 ${GAZEBO_LIBRARIES})

add_library(B2T_36_5 SHARED B2T_36_5.cc)
target_link_libraries(B2T_36_5 ${GAZEBO_LIBRARIES})

add_library(B2T_15_6 SHARED B2T_15_6.cc)
target_link_libraries(B2T_15_6 ${GAZEBO_LIBRARIES})

add_library(B2T_61_7 SHARED B2T_61_7.cc)
target_link_libraries(B2T_61_7 ${GAZEBO_LIBRARIES})

add_library(B2T_91_8 SHARED B2T_91_8.cc)
target_link_libraries(B2T_91_8 ${GAZEBO_LIBRARIES})

add_library(B2T_41_9 SHARED B2T_41_9.cc)
target_link_libraries(B2T_41_9 ${GAZEBO_LIBRARIES})

add_library(B2T_81_10 SHARED B2T_81_10.cc)
target_link_libraries(B2T_81_10 ${GAZEBO_LIBRARIES})

add_library(B2T_91_11 SHARED B2T_91_11.cc)
target_link_libraries(B2T_91_11 ${GAZEBO_LIBRARIES})

add_library(B2T_66_12 SHARED B2T_66_12.cc)
target_link_libraries(B2T_66_12 ${GAZEBO_LIBRARIES})

add_library(B2T_30_13 SHARED B2T_30_13.cc)
target_link_libraries(B2T_30_13 ${GAZEBO_LIBRARIES})

add_library(B2T_54_14 SHARED B2T_54_14.cc)
target_link_libraries(B2T_54_14 ${GAZEBO_LIBRARIES})

add_library(B2T_70_15 SHARED B2T_70_15.cc)
target_link_libraries(B2T_70_15 ${GAZEBO_LIBRARIES})

add_library(B2T_47_16 SHARED B2T_47_16.cc)
target_link_libraries(B2T_47_16 ${GAZEBO_LIBRARIES})

add_library(B2T_12_17 SHARED B2T_12_17.cc)
target_link_libraries(B2T_12_17 ${GAZEBO_LIBRARIES})

add_library(B2T_89_18 SHARED B2T_89_18.cc)
target_link_libraries(B2T_89_18 ${GAZEBO_LIBRARIES})

add_library(B2T_49_19 SHARED B2T_49_19.cc)
target_link_libraries(B2T_49_19 ${GAZEBO_LIBRARIES})
