import scala.collection.mutable.ListBuffer

/**
  * Created by spirit on 16-3-21.
  */

object eightQueue{

  type Postion = (Int, Int)
  type ListPostion = List[Postion]

  def canEat(posa:Postion, posb:Postion) = {
    (posa._1 == posb._1) || (posa._2 == posb._2) || ((posa._1 - posb._1).abs == (posa._2 - posb._2).abs)
  }

  def isSafe(pos: Postion, places: ListPostion):Boolean = {
    places.forall(!canEat(_, pos))
  }

  def placeQueue(k:Int): List[ListPostion] = {

    def recPlace(curr: Int):List[ListPostion] = {
      if (curr == 0) {
        List(List())
      }
      else {
        for {currList <- recPlace(curr - 1)
             i <- 1 to k
             pos = (curr, i)
             if (isSafe(pos, currList))
        } yield pos::currList
      }
    }

    recPlace(k)
  }
}

println(eightQueue.placeQueue(4).mkString("\n"))