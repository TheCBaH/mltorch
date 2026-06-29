(* See vec6.mli. Representation is a 6-cell array of the component type, in Axis
   order (index = Axis.to_int). The [role] parameter is phantom. *)

type 'd t = 'd array
type shape = Dim.extent Dim.t t
type coord = Dim.index Dim.t t
type deltas = Dim.delta Dim.t t

let get (v : 'd t) (a : Axis.t) : 'd = v.(Axis.to_int a)

let set (v : 'd t) (a : Axis.t) (x : 'd) : 'd t =
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

let deltas ~n ~t ~d ~h ~w ~c =
  [|
    Dim.delta n; Dim.delta t; Dim.delta d; Dim.delta h; Dim.delta w; Dim.delta c;
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

let drop_leading pred axes =
  let rec aux = function
    | [] -> [ List.nth axes (List.length axes - 1) ]
    | a :: rest -> if pred a then aux rest else a :: rest
  in
  aux axes

let pp_shape fmt (s : shape) =
  let axes = drop_leading (fun a -> Dim.to_int (get s a) = 1) Axis.all in
  Format.pp_print_char fmt '[';
  List.iteri
    (fun i a ->
      if i > 0 then Format.pp_print_char fmt ' ';
      Format.fprintf fmt "%a=%a" Axis.pp a Dim.pp (get s a))
    axes;
  Format.pp_print_char fmt ']'

let pp_coord fmt (c : coord) =
  let axes = drop_leading (fun a -> Dim.to_int (get c a) = 0) Axis.all in
  Format.pp_print_char fmt '(';
  List.iteri
    (fun i a ->
      if i > 0 then Format.pp_print_char fmt ',';
      Dim.pp fmt (get c a))
    axes;
  Format.pp_print_char fmt ')'
