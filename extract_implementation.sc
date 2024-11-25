importCode(inputPath = "./inputcode/", projectName = "test")

import java.io._
import scala.collection.mutable
import scala.io.Source
// Root path of your project
val rootPath = "inputcode/"

// Function to read from a file and extract method code
def extractMethodCodeFromFile(
    filePath: String,
    method: io.shiftleft.codepropertygraph.generated.nodes.Method): String = {

  // Read the entire file into a list of lines
  val source = scala.io.Source.fromFile(new java.io.File(filePath))
  val lines = source.getLines().toList
  source.close()

  // Use the provided line numbers to extract the method's code
  val startLine = method.lineNumber.getOrElse(throw new Exception("Start line number not provided"))
  val endLine = method.lineNumberEnd.getOrElse(throw new Exception("End line number not provided"))

  // Extract the lines corresponding to the method's body
  val methodCodeLines = lines.slice(startLine - 1, endLine)

  // Reconstruct the method code
  val methodCode = methodCodeLines.mkString("\n").trim

  // Return the extracted method code
  methodCode
}

// Function to retrieve the implementation of a method by reading from its file
def getMethodImplementationFromFile(method: Method): Option[String] = {
  val filename = method.filename

  // Filter out methods with empty, <empty>, unresolved namespaces, or operators
  if (filename.isEmpty || filename.toLowerCase.contains("empty") ||
      method.fullName.contains("<unresolvedNamespace>") ||
      method.fullName.contains("<operator>")) {
    println(s"Method ${method.fullName} has no valid associated filename.")
    return None
  }

  // Filter if the astParentFullName is global or astParentType is a namespace block
  if (method.astParentFullName == "<global>" || method.astParentType.toLowerCase.contains("namespace")) {
    println(s"Method ${method.fullName} is global or a namespace block and is skipped.")
    return None
  }

  // Append the root path to the method's filename
  val filePath = rootPath + filename

  try {
    // Extract the method code by reading from the file using the method's line range
    Some(extractMethodCodeFromFile(filePath, method))
  } catch {
    case e: Exception =>
      println(s"Error retrieving method implementation for ${method.fullName}: ${e.getMessage}")
      None
  }
}

// DFS function to recursively explore the call graph, store method implementations in a list
def dfsStoreMethods(method: Method, visited: mutable.Set[String], methods: mutable.ListBuffer[String]): Unit = {
  // If the method has been visited, skip to avoid infinite recursion
  if (visited.contains(method.fullName)) return

  // Mark the current method as visited
  visited += method.fullName

  // Get the implementation of the current method by reading from the file
  getMethodImplementationFromFile(method) match {
    case Some(implementation) =>
      methods += implementation  // Store only the method's code, no extra verbose text
    case None => // Do nothing if the method is not found or has an issue
  }

  // Get the callees (functions called by the current method)
  val callees = method.call.callee.l

  // Recursive case: go through each callee and apply DFS
  for (callee <- callees) {
    dfsStoreMethods(callee.method, visited, methods)  // Recursively call DFS for each callee
  }
}

// Initialize a BufferedWriter to write the output to a file
val outputFile = new File("call_stack_trace.txt")
val writer = new BufferedWriter(new FileWriter(outputFile))

// Call the DFS function to explore the call graph and write to the file
try {
  val visited = mutable.Set[String]()  // Create a mutable set to track visited methods
  val methods = mutable.ListBuffer[String]()  // ListBuffer to store the code of the methods

  val method = cpg.method.name("main").l(0)

  // Perform DFS starting from the root method (replace "main" with your starting method)
  dfsStoreMethods(method, visited, methods)

  val reversedMethods = methods.reverse
  for (implementation <- reversedMethods) {
    writer.write(s"$implementation\n\n")  // Write only the method code, no headers or verbose indicators
  }
} finally {
  writer.close()  // Ensure the writer is closed after writing is complete
}

println(s"Reversed call stack trace written to ${outputFile.getAbsolutePath}")

