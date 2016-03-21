import scala.collection.mutable.ArrayBuffer

/**
  * Created by spirit on 16-3-19.
  */
abstract class CalcNode{
  def value:Double
}
object CalcNode{
  class TwoNode(val op:Char, val left: CalcNode, val right: CalcNode) extends CalcNode{
    def value:Double = op match{
      case '+' => left.value + right.value
      case '-' => left.value - right.value
      case '*' => left.value * right.value
      case '/' => left.value / right.value
    }
  }
  class Number(override val value: Double) extends CalcNode
  class Mulitfy(left:CalcNode, right:CalcNode) extends TwoNode('*', left, right)
  class Devide(left:CalcNode, right:CalcNode) extends TwoNode('/', left, right)
  class Add(left:CalcNode, right:CalcNode) extends TwoNode('+', left, right)
  class Minus(left:CalcNode, right:CalcNode) extends TwoNode('-', left, right)

  def addMaker(left:CalcNode, right:CalcNode) = new Add(left, right)
  def minusMaker(left:CalcNode, right:CalcNode) = new Minus(left, right)
  def mulifyMaker(left:CalcNode, right:CalcNode) = new Mulitfy(left, right)
  def divideMaker(left:CalcNode, right:CalcNode) = new Devide(left, right)
}

object Drive{
  import CalcNode._

  def calc(str:String):Double = {
    parse(str).value
  }

  private def getToken(str:String): List[String] = {
    var temp:String = ""
    var result:ArrayBuffer[String] = new ArrayBuffer[String]()
    for (ch <- str){
      if ("()+-*/".contains(ch)) {
        if (temp != ""){
          result += temp
          temp = ""
        }
        result += ch.toString
      }
      else if ("\n\t ".contains(ch)) {
        if (temp != ""){
          result += temp
          temp = ""
        }
      }
      else {
        temp += ch.toString
      }
    }
    if (temp != "") result += temp
    result.toList
  }

  private def parse(str:String):CalcNode = {
    val tokens = getToken(str)
    exprParse(tokens)._1
  }

  private def tempParse(lowerParse:List[String]=>Tuple2[CalcNode, List[String]], nodemakers:Array[(Char, (CalcNode,CalcNode)=>CalcNode)])(tokens: List[String]):Tuple2[CalcNode, List[String]] = {

    def rec(node:CalcNode, lefTokens:List[String]): Tuple2[CalcNode, List[String]] = {
      if (lefTokens == Nil){
        return (node,lefTokens)
      }

      val h = lefTokens.head
      val rest = lefTokens.tail

      for ((ch, maker) <- nodemakers){
        if (h == ch.toString){
          val ret = lowerParse(rest)
          ret match{
            case (newNode, newlefTokens) => return rec(maker(node, newNode), newlefTokens)
          }
        }
      }

      (node, lefTokens)
    }

    val ret = lowerParse(tokens)
    ret match{
      case (newNode, newlefTokens) => rec(newNode, newlefTokens)
    }
  }

  private val termParse = tempParse(numberParse, Array(('*', mulifyMaker),('/', divideMaker)))_
  private val exprParse = tempParse(termParse, Array(('+', addMaker),('-', minusMaker)))_

  private def numberParse(tokens: List[String]):Tuple2[CalcNode, List[String]] = {
    val h = tokens.head

    val rest = tokens.tail
    h match {
      case "(" => {
        val ret = exprParse(rest)
        (ret._1, ret._2.tail) // 第一个必须为 )
      }
      case _ => ((new Number(h.toDouble)), rest)
    }
  }

}
println(Drive.calc("(3+2 )*7-8*2-(9-7)/5"))
