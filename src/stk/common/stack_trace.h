#pragma once

#include <sstream>

/*!
 * \brief Generate a stack trace.
 *
 * Generate a stack trace and store it in the given `ostream` object.
 * The top of the stack is the frame that is calling this function.
 * The format of the stack trace is platform-dependent.
 *
 * @param s    Output stream.
 * @param skip Optional number of frames to skip (counting from the
 *             top of the stack).
 */
void get_stack_trace(std::ostream& s, const int skip = 0);

