import io.shiftleft.codepropertygraph.Cpg
import io.shiftleft.codepropertygraph.generated.nodes.Method
import scala.io.Source
import java.io.{File, PrintWriter}

// Root path of your project source files
val rootPath = "inputcode/"

// Function to escape special characters for JSON
def escapeJsonString(str: String): String = {
  str.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r")
}

// Function to extract method code from a file
def extractMethodCodeFromFile(
    filePath: String,
    method: Method): String = {

  val source = Source.fromFile(filePath)
  val lines = source.getLines().toList
  source.close()

  val startLine = method.lineNumber.getOrElse(throw new Exception("Start line number not provided"))
  val endLine = method.lineNumberEnd.getOrElse(throw new Exception("End line number not provided"))

  val methodCodeLines = lines.slice(startLine - 1, endLine)
  val methodCode = methodCodeLines.mkString("\n").trim
  methodCode
}

// Function to retrieve the method implementation and details, and format it as JSON
def getMethodDetailsAsJson(method: Method, id: Int): Option[String] = {
  val filename = method.filename

  // Exclude methods with <global> in the name or parent full name
  if (filename.isEmpty || filename.toLowerCase.contains("empty") ||
      method.fullName.contains("<unresolvedNamespace>") ||
      method.fullName.contains("<operator>") ||
      // method.fullName.contains("<global>") || method.astParentFullName.contains("<global>") ||
      method.name == "<global>" ||
      method.astParentType.toLowerCase.contains("namespace")) {
    None
  } else {
    val filePath = rootPath + filename
    try {
      // Extract method code
      val methodCode = extractMethodCodeFromFile(filePath, method)

      // Format as JSON object string
      val json = s"""
      {
        "id": $id,
        "code": "${escapeJsonString(methodCode)}",
        "filename": "${escapeJsonString(filename)}",
        "fullname": "${escapeJsonString(method.fullName)}",
        "name": "${escapeJsonString(method.name)}"
      }
      """
      Some(json.trim)
    } catch {
      case e: Exception =>
        println(s"Error extracting method from $filePath: ${e.getMessage}")
        None
    }
  }
}

// Import code into Joern and create CPG (code property graph)
val cpg: Cpg = importCode(inputPath = "./inputcode/", projectName = "test")

// Extract all methods from the CPG and get them as JSON objects, excluding <global> methods
val methodDetailsList = cpg.method.toList.zipWithIndex.flatMap { case (method, index) =>
  getMethodDetailsAsJson(method, index + 1)
}

// Convert method details to a JSON-like array string
val methodDetailsJson = methodDetailsList.mkString("[\n", ",\n", "\n]")

// Save the JSON string to a file
val jsonFile = new PrintWriter(new File("./output/methods.json"))
jsonFile.write(methodDetailsJson)
jsonFile.close()

println("Methods successfully written to methods.json, excluding <global> methods.")
