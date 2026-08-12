(* See shape4.mli. Represented as the [Vec6.shape] it is, rather than as four
   separate extents: every use either forwards to shared compute or reads one
   axis, and keeping the frame representation makes [to_vec6] free and removes
   any chance of the two spellings disagreeing. The abstraction is in the
   signature, not in the runtime shape. *)

type t = Vec6.shape
type error = [ `Non_four_dimensional_shape of Vec6.shape ]

let pp_error fmt : [< error ] -> unit = function
  | `Non_four_dimensional_shape s ->
      Fmt.pf fmt "@[shape has extent on T or D: %a@]" Vec6.pp_shape s

let one = Dim.extent 1
let make ~n ~h ~w ~c = Vec6.make ~n ~t:one ~d:one ~h ~w ~c

let of_ints ~n ~h ~w ~c =
  make ~n:(Dim.extent n) ~h:(Dim.extent h) ~w:(Dim.extent w) ~c:(Dim.extent c)

let of_vec6 (s : Vec6.shape) =
  let unit_axis axis = Dim.to_int (Vec6.get s axis) = 1 in
  if unit_axis Axis.T && unit_axis Axis.D then Err.return s
  else Err.fail (`Non_four_dimensional_shape s)

let to_vec6 s = s
let get s axis = Vec6.get s (Axis4.to_axis axis)

(* Total, unlike a [Vec6.set] on T or D: [Axis4.t] cannot name those, so setting
   an axis cannot take a shape out of the dialect. *)
let set s axis v = Vec6.set s (Axis4.to_axis axis) v
let numel = Vec6.numel

(* No [Vec6] equality to delegate to; the four axes are all that can differ,
   T and D being 1 on both sides by construction. *)
let equal a b =
  List.for_all (fun axis -> Dim.equal (get a axis) (get b axis)) Axis4.all

let pp fmt s =
  Fmt.pf fmt "@[<h>[%a]@]"
    (Fmt.list ~sep:Fmt.sp (fun fmt axis ->
         Fmt.pf fmt "%a=%a" Axis4.pp axis Dim.pp (get s axis)))
    Axis4.all

let jsont : t Jsont.t =
  Jsont.map ~kind:"shape4"
    ~dec:(fun json ->
      let s = Json_util.dec Vec6.shape_jsont json in
      match of_vec6 s with
      | Ok s -> s
      | Error _ ->
          Jsont.Error.msgf Jsont.Meta.none "shape4: extent on T or D: %a"
            Vec6.pp_shape s)
    ~enc:(fun s -> Json_util.enc Vec6.shape_jsont s)
    Jsont.json
