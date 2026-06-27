(* See vec6.mli. Representation is a 6-cell array of the component type, in Axis
   order (index = Axis.to_int). The [role] parameter is phantom. *)

type (+'role, 'comp) t = 'comp array
type shape_role
type coord_role
type shape = (shape_role, Dim.extent Dim.t) t
type coord = (coord_role, Dim.index Dim.t) t

let get (v : ('role, 'comp) t) (a : Axis.t) : 'comp = v.(Axis.to_int a)

let set (v : ('role, 'comp) t) (a : Axis.t) (x : 'comp) : ('role, 'comp) t =
  let v' = Array.copy v in
  v'.(Axis.to_int a) <- x;
  v'

let shape ~n ~t ~d ~h ~w ~c =
  [|
    Dim.extent n;
    Dim.extent t;
    Dim.extent d;
    Dim.extent h;
    Dim.extent w;
    Dim.extent c;
  |]

let coord ~n ~t ~d ~h ~w ~c =
  [|
    Dim.index n; Dim.index t; Dim.index d; Dim.index h; Dim.index w; Dim.index c;
  |]

let origin = coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0

let numel (s : shape) =
  List.fold_left (fun acc a -> Dim.( *@ ) acc (get s a)) Dim.one_count Axis.all

let offset (s : shape) (c : coord) =
  List.fold_left
    (fun acc a -> Dim.lin acc (get s a) (get c a))
    Dim.zero_offset Axis.all

let in_bounds (s : shape) (c : coord) =
  (* indices are >= 0 by construction (Dim.index); just check the upper bound *)
  List.for_all (fun a -> Dim.to_int (get c a) < Dim.to_int (get s a)) Axis.all

(* extent-1 axes broadcast to index 0; others keep the coord's index. *)
let read_coord (s : shape) (c : coord) =
  let v a = if Dim.to_int (get s a) = 1 then 0 else Dim.to_int (get c a) in
  coord ~n:(v N) ~t:(v T) ~d:(v D) ~h:(v H) ~w:(v W) ~c:(v C)

let iter (s : shape) (f : coord -> unit) =
  let e a = Dim.to_int (get s a) in
  for n = 0 to e N - 1 do
    for t = 0 to e T - 1 do
      for d = 0 to e D - 1 do
        for h = 0 to e H - 1 do
          for w = 0 to e W - 1 do
            for c = 0 to e C - 1 do
              f (coord ~n ~t ~d ~h ~w ~c)
            done
          done
        done
      done
    done
  done

let pp_shape fmt (s : shape) =
  Format.fprintf fmt "[N=%a T=%a D=%a H=%a W=%a C=%a]" Dim.pp (get s N) Dim.pp
    (get s T) Dim.pp (get s D) Dim.pp (get s H) Dim.pp (get s W) Dim.pp
    (get s C)

let pp_coord fmt (c : coord) =
  Format.fprintf fmt "(%a,%a,%a,%a,%a,%a)" Dim.pp (get c N) Dim.pp (get c T)
    Dim.pp (get c D) Dim.pp (get c H) Dim.pp (get c W) Dim.pp (get c C)
