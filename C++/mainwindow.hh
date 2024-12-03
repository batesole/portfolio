// Program author
// Name: Ashley Batesole
// Student number: 150161899
// UserID: rfasba
// E-Mail: ashley.batesole@tuni.fi
//
// Instructions and other information provided in
// instructions.pdf or instructions.txt

#ifndef MAINWINDOW_HH
#define MAINWINDOW_HH

#include "gameboard.hh"
#include <QMainWindow>
#include <QPushButton>
#include <QTextBrowser>
#include <vector>
#include <QLabel>
#include <QLineEdit>
#include <QTimer>
#include <QLCDNumber>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QFont>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class Gameboard;
class Square;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void handle_character_clicks();


private:
    Ui::MainWindow *ui;

    //game board object
    GameBoard board_;

    //variables to use for win/loss scenarios
    bool win_ = false;
    bool loss_ = false;
    QLabel* gameOver = new QLabel(this);
    QFont gameOverFont = QFont("Helvetica", 16);
    const int MARGINX_GAMEOVER = 20;
    const int MARGINY_GAMEOVER = 0;
    const int GAMEOVER_WIDTH = 600;


//SET UP FOR THE BUTTON GRID--------------------------------------------------
    //starting point of button grid
    const int MARGINX_GRID = 20;
    const int MARGINY_GRID = 30;

    // Space between buttons, both horizontally and vertically
    const int BUTTON_SPACING = 5;

    // Dimensions for buttons on gameboard
    const int BUTTON_WIDTH = 30;
    const int BUTTON_HEIGTH = 30;

    //size of the gameboard, initialized to ten
    //can be changed by the user
    int boardSide = 10;

    //images to use on the buttons
    std::string flagColoredFile = ":/flag_colored.png";
    QPixmap flagColored = (QString::fromStdString(flagColoredFile));
    std::string mineImageFile = ":/mine.png";
    QPixmap mineImage = (QString::fromStdString(mineImageFile));
    std::string zeroImageFile = ":/zero.png";
    QPixmap zeroImage = (QString::fromStdString(zeroImageFile));
    std::string oneImageFile = ":/one.png";
    QPixmap oneImage = (QString::fromStdString(oneImageFile));
    std::string twoImageFile = ":/two.png";
    QPixmap twoImage = (QString::fromStdString(twoImageFile));
    std::string threeImageFile = ":/three.png";
    QPixmap threeImage = (QString::fromStdString(threeImageFile));
    std::string fourImageFile = ":/four.png";
    QPixmap fourImage = (QString::fromStdString(fourImageFile));
    std::string fiveImageFile = ":/five.png";
    QPixmap fiveImage = (QString::fromStdString(fiveImageFile));
    std::string sixImageFile = ":/six.png";
    QPixmap sixImage = (QString::fromStdString(sixImageFile));
    std::string sevenImageFile = ":/seven.png";
    QPixmap sevenImage = (QString::fromStdString(sevenImageFile));
    std::string eightImageFile = ":/eight.png";
    QPixmap eightImage = (QString::fromStdString(eightImageFile));

    //initialize the gameboard with pushbuttons
    std::vector<QPushButton*> buttons_;
    void init_buttons();

    //update the button(s) as necessary when one is clicked
    void update_buttons(int& x, int& y);
    QPixmap get_image(int& x, int& y);


//SET UP FOR THE OPTIONS GRID-------------------------------------------------
    //other game options:
    //seed value, adjust size of grid, button to place flag,
    //reset button, timer

    //starting corner of the user options layout
    int MARGINX_OPTIONS;
    int MARGINY_OPTIONS = MARGINY_GRID;

    //location of all elements in the user options layout
    //different element types have 1 marginy spacing between them
    //these are variable so they can change size with the gameboard
    int MARGINY_OPTIONS_SEED1 = MARGINY_OPTIONS * 1;
    int MARGINY_OPTIONS_SEED2 = MARGINY_OPTIONS * 2;
    int MARGINY_OPTIONS_SEED3 = MARGINY_OPTIONS * 3;
    int MARGINY_OPTIONS_BOARD1 = MARGINY_OPTIONS * 5;
    int MARGINY_OPTIONS_BOARD2 = MARGINY_OPTIONS * 6;
    int MARGINY_OPTIONS_BOARD3 = MARGINY_OPTIONS * 7;
    int MARGINY_OPTIONS_FLAG = MARGINY_OPTIONS * 9;
    int MARGINY_OPTIONS_TIMER = MARGINY_OPTIONS * 11;
    int MARGINY_OPTIONS_RESET = MARGINY_OPTIONS * 13;

    //object dimensions
    const int OPTIONS_WIDTH = 70;
    const int OPTIONS_HEIGTH = 30;
    const int OPTIONS_WIDTH_LABEL = 150;


    //seed value:
    //use a spinbox to get the value
    //update the value and gameboard once update button is clicked
    int seed_ = 0;
    QSpinBox* seedSpinBox;
    QLabel* seedSpinBoxLabel;
    QPushButton* updateSeed;
    int seedMin = 0;
    int seedMax = 9999;
    void on_update_seed();


    //gameboard size:
    //use a spinbox to get the value
    //update the value and gameboard once update button is clicked
    QSpinBox* boardSize;
    QLabel* boardSizeLabel;
    QPushButton* updateBoard;
    int boardSizeMin = 1;
    int boardSizeMax = 16;
    void on_update_board();


    //place flag button:
    //switch between placing a flag or selecting a square
    QPushButton* flagButton;
    std::string flagSelectedFile = ":/flag_selected.png";
    QPixmap flagSelected = (QString::fromStdString(flagSelectedFile));
    std::string mouseSelectedFile = ":/mouse_selected.png";
    QPixmap mouseSelected = (QString::fromStdString(mouseSelectedFile));
    bool placeFlag_ = false;
    void on_flag_button_clicked();


    //reset button:
    //reset the gameboard and the timer
    QPushButton* resetButton;
    void on_reset_button_clicked();


    //timer:
    //start once the user clicks a square
    //display on LCD Number
    QTimer* timer;
    QLCDNumber* timeDisplay;
    int seconds_ = 0;
    void on_update_time();


    //initialize user options layout
    void init_layout();

};
#endif // MAINWINDOW_HH
