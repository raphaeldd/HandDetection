#include "SaveDetections.h"

SaveDetections::SaveDetections(QString filenameAvi)
{
    this->debtCounter = 0;
    file = new QFile("Detections.xml");
    file->open(QIODevice::WriteOnly);
    xmlWriter = new QXmlStreamWriter(file);
    xmlWriter->setAutoFormatting(true);
    xmlWriter->writeStartDocument();
    xmlWriter->writeEndDocument();
    xmlWriter->writeStartElement("Saves");
    xmlWriter->writeTextElement("File", filenameAvi);
}

void SaveDetections::newFrame(int frameNr)
{
    // Depth = 1
    for(int i = this->debtCounter; i >= 1; i--) {
        xmlWriter->writeEndElement();
    }

    this->debtCounter = 1;
    xmlWriter->writeStartElement("Frame");
    xmlWriter->writeTextElement("Number", QString::number(frameNr));
}

void SaveDetections::newUpperBody(Rect box)
{
    // Dept = 2
    for(int i = this->debtCounter; i >= 2; i--) {
        xmlWriter->writeEndElement();
    }

    this->debtCounter = 2;
    xmlWriter->writeStartElement("UpperBody");
    xmlWriter->writeStartElement("Rect");
    xmlWriter->writeTextElement("x", QString::number(box.x));
    xmlWriter->writeTextElement("y", QString::number(box.y));
    xmlWriter->writeTextElement("Width", QString::number(box.width));
    xmlWriter->writeTextElement("Height", QString::number(box.height));
    xmlWriter->writeEndElement();
}

void SaveDetections::newHand(RotatedRect box)
{
    // Dept = 3
    for(int i = this->debtCounter; i >= 3; i--) {
        xmlWriter->writeEndElement();
    }

    this->debtCounter = 3;
    xmlWriter->writeStartElement("Hand");
    xmlWriter->writeStartElement("Rect");
    xmlWriter->writeTextElement("x", QString::number(box.center.x));
    xmlWriter->writeTextElement("y", QString::number(box.center.y));
    xmlWriter->writeTextElement("Width", QString::number(box.size.width));
    xmlWriter->writeTextElement("Height", QString::number(box.size.height));
    xmlWriter->writeEndElement();
    xmlWriter->writeTextElement("Rotation", QString::number(box.angle));
}

void SaveDetections::runtime(float time)
{
    this->debtCounter--;
    xmlWriter->writeEndElement();
    xmlWriter->writeTextElement("Runtime", QString::number(time));
}


SaveDetections::~SaveDetections()
{
    for(int i = this->debtCounter; i >= 0; i--) {
        xmlWriter->writeEndElement();
    }

    xmlWriter->writeEndDocument();
    file->close();
}
