// Program author
// Name: Ashley Batesole
// Student number: 150161899
// UserID: rfasba
// E-Mail: ashley.batesole@tuni.fi
//
// Instructions and other information provided in
// instructions.pdf or instructions.txt

#include "mainwindow.hh"
#include "square.hh"
#include "gameboard.hh"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
