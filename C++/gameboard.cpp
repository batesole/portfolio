#include "gameboard.hh"
#include <random>

GameBoard::GameBoard()
{
}

GameBoard::~GameBoard()
{
}

void GameBoard::init(int seed, int& boardSide)
{
    BOARD_SIDE = boardSide;
    std::default_random_engine randomEng(seed);
    std::uniform_int_distribution<int> distribution(1, 100/MINE_PROBABALITY);

    for(int y = 0; y < BOARD_SIDE; ++y)
    {
        std::vector<Square> line;
        for(int x = 0; x < BOARD_SIDE; ++x)
        {
            bool hasMine = false;
            if(distribution(randomEng) == 1)
            {
                hasMine = true;
            }
            Square new_square(x, y, hasMine, this);
            line.push_back(new_square);
        }
        board_.push_back(line);
    }

    // When all the squares of the board exist, each square is asked to
    // calculate the data telling how many mines lie in the adjacent squares.
    // After that game board is ready.
    for(int y = 0; y < BOARD_SIDE; ++y)
    {
        for(int x = 0; x < BOARD_SIDE; ++x)
        {
            board_.at(y).at(x).countAdjacent();
        }
    }
}


bool GameBoard::isGameOver() const
{
    for(int y = 0; y < BOARD_SIDE; ++y)
    {
        for(int x = 0; x < BOARD_SIDE; ++x)
        {
            if(not board_.at(y).at(x).isReady())
            {
                return false;
            }
        }
    }
    return true;
}

int GameBoard::getSize() const
{
    return board_.size();
}

Square GameBoard::getSquare(int x, int y) const
{
    return board_.at(y).at(x);
}

void GameBoard::setSquare(Square sq, int x, int y)
{
    board_.at(y).at(x) = sq;
}

bool GameBoard::openSquare(int x, int y)
{
    return board_.at(y).at(x).open();

}
