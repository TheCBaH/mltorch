(* Symbolic integers for PyTorch-style dynamic shapes: an axis size may be a
   variable (s0, s1, …) or an affine expression over them, carried with a
   ShapeEnv of per-variable constraints (range + divisibility). Direct execution
   binds the variables to concrete ints (checking the guards) before allocating;
   the symbolic/footprint layers keep them symbolic. See native_tensor_design §4. *)

type t = Const of int | Var of string | Add of t * t | Scale of int * t

let const n = Const n
let var name = Var name
let add a b = Add (a, b)
let scale k a = Scale (k, a)

let rec pp fmt = function
  | Const n -> Format.fprintf fmt "%d" n
  | Var v -> Format.pp_print_string fmt v
  | Add (a, b) -> Format.fprintf fmt "%a + %a" pp a pp b
  | Scale (k, a) -> Format.fprintf fmt "%d*%a" k pp a

(* A variable's declared constraints. [divisor = 1] means "no divisibility
   requirement"; [hi] is exclusive. *)
type var_info = { lo : int; hi : int; divisor : int }
type env = { decls : (string * var_info) list; binds : (string * int) list }

let empty = { decls = []; binds = [] }

let declare env name ~lo ~hi ?(divisor = 1) () =
  { env with decls = (name, { lo; hi; divisor }) :: env.decls }

(* Bind a variable to a concrete value, checking it against the declared
   constraints (mirrors dynamo's guard check). Raises [Invalid_argument]. *)
let bind env name value =
  (match List.assoc_opt name env.decls with
  | Some { lo; hi; divisor } ->
      if value < lo || value >= hi then
        invalid_arg
          (Printf.sprintf "Symint.bind: %s=%d out of [%d,%d)" name value lo hi);
      if value mod divisor <> 0 then
        invalid_arg
          (Printf.sprintf "Symint.bind: %s=%d not divisible by %d" name value
             divisor)
  | None -> ());
  { env with binds = (name, value) :: env.binds }

(* [Some n] once every variable in the expression is bound. *)
let rec eval env = function
  | Const n -> Some n
  | Var v -> List.assoc_opt v env.binds
  | Add (a, b) -> (
      match (eval env a, eval env b) with
      | Some x, Some y -> Some (x + y)
      | _ -> None)
  | Scale (k, a) -> (
      match eval env a with Some x -> Some (k * x) | None -> None)

(* An axis size: concrete, or a symbolic expression awaiting binding. *)
type size = Static of int | Sym of t

let pp_size fmt = function
  | Static n -> Format.fprintf fmt "%d" n
  | Sym s -> pp fmt s

let resolve env = function Static n -> Some n | Sym s -> eval env s
