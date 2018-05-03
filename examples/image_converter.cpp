#include <stk/image/volume.h>
#include <stk/io/io.h>

#include <iostream>

using namespace stk;

void print_help(const char* name)
{
    std::cout << "Usage: " << name << " <in> <out>" << std::endl;
}

int main(int argc, char* argv[])
{
    if (argc != 3) {
        print_help(argv[0]);
        return 1;
    }

    Volume in = read_volume(argv[1]);
    if (!in.valid()) {
        return 1;
    }

    write_volume(argv[2], in);
    return 0;
}