import io.shiftleft.codepropertygraph.Cpg
import io.shiftleft.codepropertygraph.generated.nodes.Method
import scala.io.Source
import java.io.{File, PrintWriter}
import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.write

// Root path of your project source files
val rootPath = "inputcode/"

// Enable default JSON formats for case classes
implicit val formats: Formats = Serialization.formats(NoTypeHints)

// Case class to represent method details
case class MethodDetails(id: Int, code: String, filename: String, fullname: String, name: String)

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

// Function to retrieve the method implementation and details as a case class
def getMethodDetailsAsJson(method: Method, id: Int): Option[MethodDetails] = {
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
      Some(MethodDetails(id, methodCode, filename, method.fullName, method.name))
    } catch {
      case e: Exception =>
        println(s"Error extracting method from $filePath: ${e.getMessage}")
        None
    }
  }
}

// Import code into Joern and create CPG (code property graph)
val cpg: Cpg = importCode(inputPath = "./inputcode/", projectName = "test")

// Reindex the valid methods, excluding <global> methods
var validId = 1
val methodDetailsList = cpg.method.toList.flatMap { method =>
  getMethodDetailsAsJson(method, validId) match {
    case Some(details) =>
      validId += 1 // Increment valid ID only for valid methods
      Some(details)
    case None =>
      None
  }
}

// Convert method details to JSON and save to file using json4s
val jsonString = write(methodDetailsList)
val jsonFile = new PrintWriter(new File("./output/methods.json"))
jsonFile.write(jsonString)
jsonFile.close()

println("Methods successfully written to methods.json using json4s, excluding <global> methods and reindexed.")
