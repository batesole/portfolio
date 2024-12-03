// Program author
// Name: Ashley Batesole
// Student number: 150161899
// UserID: rfasba
// E-Mail: ashley.batesole@tuni.fi
//
// Instructions and other information provided in
// instructions.pdf or instructions.txt

#include "mainwindow.hh"
#include "ui_mainwindow.h"
#include <QGridLayout>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //label to be used for a game over
//    gameOver = new QLabel(this);

    //make the gameboard before making the button layout
    board_.init(seed_, boardSide);

    //initialize the buttons and options layout
    init_buttons();
    init_layout();
}

MainWindow::~MainWindow()
{
    delete ui;
}

//function to handle when user clicks a button on gameboard
void MainWindow::handle_character_clicks()
{
    //start the timer when a button is clicked for the first time
    //don't restart the counter once it's running
    //that prevents button spamming to stop the timer
    if (timer != nullptr and !timer->isActive()){
        timer->start(1000);
    }

    //convert the button in the vector to the x,y location on the gameboard
    int x = 0;
    int y = 0;

    for (auto& button : buttons_)
    {      
        if(x == boardSide){
            ++y;
            x = 0;
        }

        //iterate through the buttons to find the one sending the signal
        if (button == sender())
        {
            update_buttons(x, y);

            //only one button can be clicked at a time
            break;
        }

        //increase the x coordinate to match which button we'll be looking at
        ++x;
    }

    if((board_.isGameOver() and not loss_) or win_)
    {
        //give some sort of YOU WON
        win_ = true;
        gameOver->setGeometry(MARGINX_GAMEOVER, MARGINY_GAMEOVER, GAMEOVER_WIDTH, OPTIONS_HEIGTH);
        gameOver->setText("CONGRATULATIONS ON YOUR GLORIOUS TRIUMPH");
        gameOver->setStyleSheet("color: blue");
        gameOver->setFont(gameOverFont);
        gameOver->show();

        //stop the timer
        timer->stop();

        //disable all buttons on the grid
        for(auto button : buttons_)
        {
            button->setDisabled(true);
        }
    }
}

//initialize the gameboard as an array of buttons
void MainWindow::init_buttons()
{
    //creating grid of buttons in the button vector
    for(int iy = 0; iy < boardSide; iy++){
        for(int ix = 0; ix < boardSide; ix++){

            QPushButton* pushButton = new QPushButton("", this);

            pushButton->setGeometry(MARGINX_GRID + ix * (BUTTON_WIDTH +
                                                         BUTTON_SPACING),
                                    MARGINY_GRID + iy * (BUTTON_WIDTH +
                                                         BUTTON_SPACING),
                                    BUTTON_WIDTH,
                                    BUTTON_HEIGTH);
            pushButton->setStyleSheet("background-color: blue");
            buttons_.push_back(pushButton);

            connect(pushButton, &QPushButton::clicked,
                    this, &MainWindow::handle_character_clicks);

            pushButton->show();

        }
    }
}

//update the buttons display
void MainWindow::update_buttons(int& x, int& y)
{
    //get the square's information from the given coordinates
    Square square = board_.getSquare(x,y);
    int vectorElement = x + boardSide*y;

    //update the square status
    if(square.hasFlag())
    {
        if(placeFlag_)
        {
            square.removeFlag();

            //removal of the flag icon is done here for sake of simplicity
            buttons_[vectorElement]->setIcon(QIcon());
        }
    }
    else if(not placeFlag_)
    {
        if(not square.open())
        {
            //user has opened a mine and lost
            loss_ = true;
            gameOver->setGeometry(MARGINX_GAMEOVER, MARGINY_GAMEOVER, GAMEOVER_WIDTH, OPTIONS_HEIGTH);
            gameOver->setText("Sad Days Are Here, For You Have Lost ='(");
            gameOver->setStyleSheet("color: red");
            gameOver->setFont(gameOverFont);
            gameOver->show();

            //stop the timer
            timer->stop();

            //show the mine
            buttons_[vectorElement]->setIcon(mineImage);

            //disable all buttons on the grid
            for(auto button : buttons_)
            {
                button->setDisabled(true);
            }
        }
    }
    else
    {
        square.addFlag();
    }

    //copy the modifications done to the square onto the gameboard object
    board_.setSquare(square, x, y);

    //check if there are any unopened squares without a mine
    //the user has won if not
    bool haswon = true;

    //iterate through the gameboard and update the buttons for the
    //squares that have been opened and have flags
    for(int iy = 0; iy < boardSide; ++iy)
    {
        for(int ix = 0; ix < boardSide; ++ix)
        {
            //if the square is open update the button
            if(board_.board_.at(ix).at(iy).isOpen())
            {
                QPixmap Icon = get_image(ix, iy);

                //update the button located at those coordinates with a number
                vectorElement = iy + boardSide*ix;
                buttons_[vectorElement]->setIcon(Icon);
                buttons_[vectorElement]->setIconSize(QSize(BUTTON_WIDTH,
                                                           BUTTON_HEIGTH));
                buttons_[vectorElement]->setDisabled(true);
            }

            //if the square is closed but does not have a mine the game is
            //not over yet
            else if(!board_.board_.at(ix).at(iy).hasMine())
            {
                haswon = false;
            }

            //if the square has a flag update the button
            if(board_.board_.at(ix).at(iy).hasFlag())
            {
                //update the button located at those coordinates with a flag
                vectorElement = iy + boardSide*ix;
                buttons_[vectorElement]->setIcon(flagColored);
                buttons_[vectorElement]->setIconSize(QSize(BUTTON_WIDTH,
                                                           BUTTON_HEIGTH));
            }
        }
    }

    //if the user has won through opening all buttons that aren't mines then
    //open all the mines and declare a victory
    if(haswon)
    {
        win_ = true;

        for(int iy = 0; iy < boardSide; ++iy)
        {
            for(int ix = 0; ix < boardSide; ++ix)
            {
                if(!board_.board_.at(ix).at(iy).isOpen())
                {
                    QPixmap Icon = mineImage;

                    vectorElement = iy + boardSide*ix;
                    buttons_[vectorElement]->setIcon(Icon);
                    buttons_[vectorElement]->setIconSize(QSize(BUTTON_WIDTH,
                                                               BUTTON_HEIGTH));
                    buttons_[vectorElement]->setDisabled(true);
                }
            }

        }
    }
}

//return an image based on the value in the board game square
QPixmap MainWindow::get_image(int& x, int& y)
{
    if(board_.board_.at(x).at(y).getValue() == "0")
    {
        return zeroImage;
    }
    else if(board_.board_.at(x).at(y).getValue() == "1")
    {
        return oneImage;
    }
    else if(board_.board_.at(x).at(y).getValue() == "2")
    {
        return twoImage;
    }
    else if(board_.board_.at(x).at(y).getValue() == "3")
    {
        return threeImage;
    }
    else if(board_.board_.at(x).at(y).getValue() == "4")
    {
        return fourImage;
    }
    else if(board_.board_.at(x).at(y).getValue() == "5")
    {
        return fiveImage;
    }
    else if(board_.board_.at(x).at(y).getValue() == "6")
    {
        return sixImage;
    }
    else if(board_.board_.at(x).at(y).getValue() == "7")
    {
        return sevenImage;
    }
    else if(board_.board_.at(x).at(y).getValue() == "8")
    {
        return eightImage;
    }
    else
    {
        return mineImage;
    }

}


void MainWindow::on_update_seed()
{
    //update the seed and remake the game board
    seed_ = seedSpinBox->value();

    //reset the timer
    if(timer != nullptr){
        seconds_ = 0;
        timer->stop();
        timeDisplay->display(seconds_);
    }

    //reset win/loss results
    win_ = false;
    loss_ = false;
    gameOver->setHidden(true);

    //delete the current board
    for(auto buttons : buttons_){
        buttons->close();
    }
    buttons_.clear();
    board_.board_.clear();

    //remake the game board
    board_.init(seed_, boardSide);
    init_buttons();
}

//update the game board size when the update button is clicked
void MainWindow::on_update_board()
{
    boardSide = boardSize->value();

    //reset the timer
    if(timer != nullptr){
        seconds_ = 0;
        timer->stop();
        timeDisplay->display(seconds_);
    }

    //reset win/loss results
    win_ = false;
    loss_ = false;
    gameOver->setHidden(true);

    //delete the current board
    for(auto buttons : buttons_){
        buttons->close();
    }
    buttons_.clear();
    board_.board_.clear();

    //delete the options layout and remake it
    seedSpinBox->close();
    seedSpinBoxLabel->close();
    updateSeed->close();
    boardSize->close();
    boardSizeLabel->close();
    updateBoard->close();
    flagButton->close();
    resetButton->close();
    timeDisplay->close();

    init_layout();

    //remake the game board
    board_.init(seed_, boardSide);
    init_buttons();
}

//when the flag button is clicked update placeFlag_ and the button icon
void MainWindow::on_flag_button_clicked()
{
    if(placeFlag_ == false){
        placeFlag_ = true;
        flagSelected = flagSelected.scaled(OPTIONS_WIDTH, OPTIONS_HEIGTH,
                                           Qt::KeepAspectRatio);
        flagButton->setIcon(flagSelected);
    }
    else{
        placeFlag_ = false;
        mouseSelected = mouseSelected.scaled(OPTIONS_WIDTH, OPTIONS_HEIGTH,
                                             Qt::KeepAspectRatio);
        flagButton->setIcon(mouseSelected);
    }
}

//when the reset button is clicked reset the gameboard and timer
void MainWindow::on_reset_button_clicked()
{
    //reset the timer
    if(timer != nullptr){
        seconds_ = 0;
        timer->stop();
        timeDisplay->display(seconds_);
    }

    //reset win/loss results
    win_ = false;
    loss_ = false;
    gameOver->setHidden(true);

    //delete the current board
    for(auto buttons : buttons_){
        buttons->close();
    }
    buttons_.clear();
    board_.board_.clear();

    //remake the game board
    board_.init(seed_, boardSide);
    init_buttons();
}

//initialize the layout of the other user options
//seed value, size of gameboorad, flag/click button, timer, reset button
void MainWindow::init_layout()
{
    //set the starting point of the options grid layout based on the size of
    //the gameboard
    MARGINX_OPTIONS = MARGINX_GRID + (boardSide + 1) * (BUTTON_WIDTH +
                                                         BUTTON_SPACING);

    //create the seed value input (spinbox)
    seedSpinBox = new QSpinBox(this);
    seedSpinBox->setGeometry(MARGINX_OPTIONS,MARGINY_OPTIONS_SEED2,
                             OPTIONS_WIDTH, OPTIONS_HEIGTH);
    seedSpinBox->setMinimum(seedMin);
    seedSpinBox->setMaximum(seedMax);
    seedSpinBox->show();

    //text for the seed line editor
    seedSpinBoxLabel = new QLabel(this);
    seedSpinBoxLabel->setGeometry(MARGINX_OPTIONS,MARGINY_OPTIONS_SEED1,
                                  OPTIONS_WIDTH_LABEL, OPTIONS_HEIGTH);
    seedSpinBoxLabel->setText("Seed Value:");
    seedSpinBoxLabel->show();

    //button to update the seed value
    updateSeed = new QPushButton("Update", this);
    updateSeed->setGeometry(MARGINX_OPTIONS, MARGINY_OPTIONS_SEED3,
                            OPTIONS_WIDTH, OPTIONS_HEIGTH);
    connect(updateSeed, &QPushButton::clicked,
            this, &MainWindow::on_update_seed);
    updateSeed->show();

    //create the board size input (spinbox)
    boardSize = new QSpinBox(this);
    boardSize->setGeometry(MARGINX_OPTIONS, MARGINY_OPTIONS_BOARD2,
                           OPTIONS_WIDTH, OPTIONS_HEIGTH);
    boardSize->setMinimum(boardSizeMin);
    boardSize->setMaximum(boardSizeMax);
    boardSize->setValue(boardSide);
    boardSize->show();

    //text for the board size line editor
    boardSizeLabel = new QLabel(this);
    boardSizeLabel->setGeometry(MARGINX_OPTIONS, MARGINY_OPTIONS_BOARD1,
                                OPTIONS_WIDTH_LABEL, OPTIONS_HEIGTH);
    boardSizeLabel->setText("Game Board Size:");
    boardSizeLabel->show();

    //button to update board size
    updateBoard = new QPushButton("Update", this);
    updateBoard->setGeometry(MARGINX_OPTIONS, MARGINY_OPTIONS_BOARD3,
                             OPTIONS_WIDTH, OPTIONS_HEIGTH);
    connect(updateBoard, &QPushButton::clicked,
            this, &MainWindow::on_update_board);
    updateBoard->show();

    //create the flag button
    flagButton = new QPushButton(this);
    flagButton->setGeometry(MARGINX_OPTIONS, MARGINY_OPTIONS_FLAG,
                            OPTIONS_WIDTH, OPTIONS_HEIGTH);
    connect(flagButton, &QPushButton::clicked,
            this, &MainWindow::on_flag_button_clicked);
    flagButton->show();
    flagButton->setIcon(mouseSelected);
    flagButton->setIconSize(QSize(OPTIONS_WIDTH, OPTIONS_HEIGTH));
    placeFlag_ = false;

    //create a timer and display it in an lcd display
    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &MainWindow::on_update_time);

    timeDisplay = new QLCDNumber(this);
    timeDisplay->setGeometry(MARGINX_OPTIONS, MARGINY_OPTIONS_TIMER,
                             OPTIONS_WIDTH, OPTIONS_HEIGTH);
    timeDisplay->show();

    //create the reset button
    resetButton = new QPushButton("reset", this);
    resetButton->setGeometry(MARGINX_OPTIONS, MARGINY_OPTIONS_RESET,
                             OPTIONS_WIDTH, OPTIONS_HEIGTH);
    connect(resetButton, &QPushButton::clicked,
            this, &MainWindow::on_reset_button_clicked);
    resetButton->show();
}

//update the LCD display every second
void MainWindow::on_update_time()
{
    seconds_++;

    timeDisplay->display(seconds_);
}

