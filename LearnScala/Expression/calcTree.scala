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
}

object Drive{
  import CalcNode._

  def calc(str:String):Double = {
    parse(str).value
  }

  private def getToken(str:String): List[String] = str.split(' ').toList


  private def parse(str:String):CalcNode = {
    val tokens = getToken(str)
    exprParse(tokens)._1
  }

  private def exprParse(tokens: List[String]):Tuple2[CalcNode, List[String]] = {

    def getExprAddMinus(node:CalcNode, lefTokens:List[String]): Tuple2[CalcNode, List[String]] = {
      lefTokens match {
        case "+"::rest => {
          val ret = termParse(rest)
          ret match{
            case (newNode, newlefTokens) => getExprAddMinus(new Add(node, newNode), newlefTokens)
          }
        }
        case "-"::rest => {
          val ret = termParse(rest)
          ret match{
            case (newNode, newlefTokens) => getExprAddMinus(new Minus(node, newNode), newlefTokens)
          }
        }
        case _ => (node, lefTokens)
      }
    }

    val ret = termParse(tokens)
    ret match{
      case (newNode, newlefTokens) => getExprAddMinus(newNode, newlefTokens)
    }
  }

  private def termParse(tokens: List[String]):Tuple2[CalcNode, List[String]] = {

    def getExprMultfyDivid(node:CalcNode, lefTokens:List[String]): Tuple2[CalcNode, List[String]] = {
      lefTokens match {
        case "*"::rest => {
          val ret = numberParse(rest)
          ret match{
            case (newNode, newlefTokens) => getExprMultfyDivid(new Mulitfy(node, newNode), newlefTokens)
          }
        }
        case "/"::rest => {
          val ret = numberParse(rest)
          ret match{
            case (newNode, newlefTokens) => getExprMultfyDivid(new Devide(node, newNode), newlefTokens)
          }
        }
        case _ => (node, lefTokens)
      }
    }

    val ret = numberParse(tokens)
    ret match{
      case (newNode, newlefTokens) => getExprMultfyDivid(newNode, newlefTokens)
    }
  }

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
println(Drive.calc("( 3 + 2 ) * 7 - 8 * 2"))
