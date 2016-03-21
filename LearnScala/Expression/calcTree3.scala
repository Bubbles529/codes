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
  class UnNode(val op:Char, val node:CalcNode) extends CalcNode {
    def value: Double = op match {
      case '+' => node.value
      case '-' => -node.value
    }
  }
  class PosNum(node:CalcNode) extends UnNode('+', node)
  class NegNum(node:CalcNode) extends UnNode('-', node)

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

object Parse{

  def parse(str:String):CalcNode = {
    tokens = getToken(str)
    exprParse(0)._1
  }

  type Tokens = ArrayBuffer[String]
  type Token = String
  type NodeMaker = (CalcNode,CalcNode)=>CalcNode
  type ParseResult = (CalcNode, Int)
  type Parse = (Int)=> ParseResult

  var tokens:Tokens = new Tokens

  private def getToken(str:String): ArrayBuffer[String] = {
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
    result
  }

  import CalcNode._

  private def tempParse(lowerParse:Parse, nodemakers:Array[(Token, NodeMaker)])(index: Int): ParseResult = {

    var ret = lowerParse(index)
    var node = ret._1
    var newIndex = ret._2
    var newNode:CalcNode = null

    while (true) {
      if (newIndex == tokens.length) {
        return (node, newIndex)
      }

      val h = tokens(newIndex)
      var matched = false
      for ((str, maker) <- nodemakers) {
        if (h == str) {
          matched = true
          ret = lowerParse(newIndex + 1)
          newNode = ret._1
          newIndex = ret._2
          node = maker(node, newNode)
        }
      }
      if (!matched) return (node, newIndex)
    }

    (node, newIndex)
  }

  private val termParse = tempParse(numberParse, Array(("*", mulifyMaker),("/", divideMaker)))_
  private val exprParse = tempParse(termParse, Array(("+", addMaker),("-", minusMaker)))_

  private def numberParse(index:Int): ParseResult = {
    val h = tokens(index)
    h match {
      case "(" => {
        val ret = exprParse(index+1)
        (ret._1, ret._2 + 1) // 第一个必须为 )
      }
      case "-" => {
        val ret = numberParse(index+1)
        (new NegNum(ret._1), ret._2)
      }
      case "+" => {
        val ret = numberParse(index+1)
        (new PosNum(ret._1), ret._2)
      }
      case _ => ((new Number(h.toDouble)), index+1)
    }
  }
}

object Drive{

  import Parse._
  import CalcNode._

  def calc(str:String):Double = {
    parse(str).value
  }

}

println(Drive.calc("(3+-22 )*7-8*2-(9-7)/5"))
