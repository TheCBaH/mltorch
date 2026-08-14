(* The affine bias layout shared by conv2d, conv2d.padding, convolution and
   linear: [1,1,1,1,1,Cout], with Cout read off the weight's N extent.

   ONE definition, reached from both directions. The evaluators use it to
   synthesize a zero bias when the operand is absent ([Eval_op]); shape
   inference uses it to REJECT a present one whose extent disagrees. Those were
   the same rule stated once and checked nowhere: a short bias built a graph and
   then made [Tensor.read]'s strict bounds check raise an uncaught
   [Invalid_argument] partway through evaluation, at whichever output channel
   first ran off the end. *)

let shape ~(weight_shape : Vec6.shape) =
  Vec6.set
    (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
    Axis.C
    (Vec6.get weight_shape Axis.N)

(* Takes the EXPECTED shape rather than deriving it, because [weight.N] is not
   the answer for every op that has a bias. A transposed convolution's ATen
   weight is [Cin, Cout/groups, kH, kW], so its output channel count is
   [weight.C * groups] -- [Conv.Convolution.bias_shape] owns that distinction,
   and a checker that re-derived [Cout] itself would have to know about it too.
   One briefly did, and rejected every valid transposed convolution whose input
   and output channel counts differ. *)
let check ~(expected : Vec6.shape) ~(actual : Vec6.shape) :
    (unit, Shape_error.t) Err.t =
  if
    List.for_all
      (fun a -> Dim.equal (Vec6.get expected a) (Vec6.get actual a))
      Axis.all
  then Err.return ()
  else
    Err.fail
      (`Operand_shape
         Shape_error.Operand_shape.{ operand = `Bias; expected; actual })
