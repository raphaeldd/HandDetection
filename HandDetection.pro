#-------------------------------------------------
#
# Project created by QtCreator 2014-02-10T12:31:52
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = HandDetection
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

QMAKE_CXXFLAGS += -fopenmp

LIBS = -lxml2 -I/opt/local/include/libxml2/
LIBS += -L/opt/local/include/ -ljpeg
LIBS += -lfftw3f -I/opt/local/lib -L/opt/local/lib
LIBS += -fopenmp
LIBS += -lopencv_nonfree -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video -lopencv_contrib -lopencv_objdetect -lopencv_legacy -lopencv_features2d -lopencv_ml -lopencv_flann -lopencv_calib3d -lopencv_gpu -I"/opt/local/include" -L"/opt/local/lib"

DEFINES += USE_FFTW

INCLUDEPATH = /usr/include/libxml2 \
        /usr/local/include

SOURCES += \
        Mixture.cpp \
        JPEGImage.cpp \
        Model.cpp \
        HOGPyramid.cpp \
        Patchwork.cpp \
        Rectangle.cpp \
        Scene.cpp \
        Object.cpp \
        PersonDetection.cpp \
        main.cpp \
    DetBody.cpp \
    DetHand.cpp \
    SaveDetections.cpp \
    HighestLikelihood.cpp


HEADERS += \
        Mixture.h \
        JPEGImage.h \
        Model.h \
        HOGPyramid.h \
        Patchwork.h \
        Rectangle.h \
        SimpleOpt.h \
        Scene.h \
        Intersector.h \
        Object.h \
        PersonDetection.h \
        Parameters.h \
    DetBody.h \
    DetHand.h \
    SaveDetections.h \
    HighestLikelihood.h

