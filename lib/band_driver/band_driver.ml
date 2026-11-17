open Ssa_ir
module B = Ssa_builder
module S = Band_schedule

module Stage = struct
  type t = { kernel : S.kernel; cin : int64; cout : int64 }
end

type mutation = Iterations_short | Step_dropped | Window_offset_dropped
type refusal = Unsupported_window of string

let pp_refusal ppf = function
  | Unsupported_window k ->
      Fmt.pf ppf
        "%s is not a stride 1 kernel without padding or dilation (the rows a \
         window reads are not consecutive producer rows)"
        k

(* A row of a tensor as the driver addresses it: a row of a whole tensor, or a
   row of a window's ring. *)
module Position = struct
  type t = Logical of S.Row.t | Ring of S.Ring_row.t

  (* The element offset of the row's first cell in a tensor of [width] pixels
     of [channels]. *)
  let element t ~width ~channels =
    let row =
      match t with Logical r -> S.Row.to_int r | Ring r -> S.Ring_row.to_int r
    in
    Int64.mul (Int64.mul (Int64.of_int row) width) channels
end

module Loop = struct
  type t = { first : S.Count.t; period : S.Count.t; iterations : S.Count.t }
end

module Program = struct
  type t = {
    program : Ssa_program.t;
    input : Ssa_id.Buffer.t;
    weights : Ssa_id.Buffer.t list;
    output : Ssa_id.Buffer.t;
    rings : (Ssa_id.Buffer.t * S.Count.t) list;
    loop : Loop.t option;
  }
end

let id = Ssa_id.Buffer.of_int
let i64 = Int64.of_int

let supported (k : S.kernel) =
  S.Count.to_int k.S.stride = 1
  && S.Count.to_int k.S.pad = 0
  && S.Count.to_int k.S.dilation = 1

let extent (st : Stage.t) = i64 (S.Count.to_int st.Stage.kernel.S.extent)

(* The product of one stage over [pixels] pixels of one row of its input: A is
   pixels x (extent * cin), the pixel's [extent] columns of channels being
   contiguous and the pixels [cin] apart; B one row of the weights,
   (extent * cin) x cout; C pixels x cout. A window of several rows is one such
   product per row, each past the first adding into C ([tap] is the row's
   position in the window); a pointwise stage's one row covers every pixel the
   event makes. *)
let desc (st : Stage.t) ~pixels ~tap =
  Result.get_ok
    (Err.payload
       (Contraction.make ~batch:1L ~m:pixels ~n:st.Stage.cout
          ~k:(Int64.mul (extent st) st.Stage.cin)
          ~a:{ Contraction.Operand.batch = 0L; outer = st.Stage.cin; k = 1L }
          ~b:{ Contraction.Operand.batch = 0L; outer = 1L; k = st.Stage.cout }
          ~c:{ Contraction.Output.batch = 0L; row = st.Stage.cout; col = 1L }
          ~format:Contraction.Element.F32
          ~seed:
            (if tap = 0 then Contraction.Seed.Zero else Contraction.Seed.Tensor)
          ~numerics:Contraction.Numerics.Ordered))

(* Where tap [tap]'s weights start in the stage's weight buffer. *)
let tap_weights (st : Stage.t) ~tap =
  Int64.mul
    (Int64.mul (i64 tap) (Int64.mul (extent st) st.Stage.cin))
    st.Stage.cout

(* Per stage, the width of its input and of its output, and the rows of its
   output over [rows] input rows. *)
let shapes stages ~width ~rows =
  let w = ref width and h = ref (S.Count.of_int rows) in
  Array.of_list
    (List.map
       (fun (st : Stage.t) ->
         let k = st.Stage.kernel in
         let win = !w in
         let wout = Int64.sub win (Int64.sub (extent st) 1L) in
         if Int64.compare wout 1L < 0 then
           invalid_arg "Band_driver: a window wider than its input";
         h := S.output_height k ~input:!h;
         w := wout;
         (win, wout, S.Count.to_int !h))
       stages)

let buffer n ~pixels ~channels role =
  {
    Ssa_buffer.id = id n;
    extents = Expr.Coord.make ~n:1L ~t:1L ~d:1L ~h:pixels ~w:channels ~c:1L;
    format = Ssa_format.F32;
    role;
  }

(* The buffers of a chain: the input is 0, the weights 1 to n, the
   intermediates n + 1 to 2n - 1 and the output 2n. A stage's output has the
   rows and width [shapes] gives, or [out_rows.(i)] rows where its consumer
   keeps only a window. *)
let layout stages ~width ~rows ~out_rows =
  let n = List.length stages in
  let shape = shapes stages ~width ~rows in
  let stages = Array.of_list stages in
  let first = stages.(0) in
  let input =
    buffer 0
      ~pixels:(Int64.mul (i64 rows) width)
      ~channels:first.Stage.cin Ssa_buffer.Input
  in
  let weights =
    List.init n (fun i ->
        buffer (1 + i)
          ~pixels:
            (Int64.mul
               (extent stages.(i))
               (Int64.mul (extent stages.(i)) stages.(i).Stage.cin))
          ~channels:stages.(i).Stage.cout Ssa_buffer.Input)
  in
  let outs =
    List.init n (fun i ->
        let _, wout, _ = shape.(i) in
        buffer
          (if i = n - 1 then 2 * n else n + 1 + i)
          ~pixels:(Int64.mul (i64 out_rows.(i)) wout)
          ~channels:stages.(i).Stage.cout
          (if i = n - 1 then Ssa_buffer.Output else Ssa_buffer.Scratch))
  in
  (input :: weights) @ outs

let ids bufs = List.map (fun (b : Ssa_buffer.t) -> b.Ssa_buffer.id) bufs

let whole stages ~width ~rows =
  let n = List.length stages in
  let rows = S.Count.to_int rows in
  let shape = shapes stages ~width ~rows in
  let out_rows = Array.map (fun (_, _, h) -> h) shape in
  let bufs = layout stages ~width ~rows ~out_rows in
  let all = ids bufs in
  let input = List.hd all in
  let weights = List.filteri (fun i _ -> i >= 1 && i <= n) all in
  let outs = List.filteri (fun i _ -> i > n) all in
  let built =
    Err.or_raise ~pp_error:Ssa_verify.pp_error
      (B.program ~buffers:bufs (fun bld ->
           List.iteri
             (fun i (st : Stage.t) ->
               let src = if i = 0 then input else List.nth outs (i - 1) in
               let win, wout, h = shape.(i) in
               let k = S.Count.to_int st.Stage.kernel.S.extent in
               if k = 1 then
                 B.contract bld
                   (desc st ~pixels:(Int64.mul (i64 h) wout) ~tap:0)
                   ~a:src ~b:(List.nth weights i) ~c:(List.nth outs i)
               else
                 (* a window is one product per output row and row of it,
                    over whole tensors *)
                 for r = 0 to h - 1 do
                   for tap = 0 to k - 1 do
                     let at x = B.Base (B.index bld x) in
                     B.view_call bld
                       (desc st ~pixels:wout ~tap)
                       ~a:
                         ( src,
                           at
                             (Position.element
                                (Position.Logical (S.Row.of_int (r + tap)))
                                ~width:win ~channels:st.Stage.cin) )
                       ~b:(List.nth weights i, at (tap_weights st ~tap))
                       ~c:
                         ( List.nth outs i,
                           at
                             (Position.element
                                (Position.Logical (S.Row.of_int r))
                                ~width:wout ~channels:st.Stage.cout) )
                   done
                 done)
             stages))
  in
  {
    Program.program = built;
    input;
    weights;
    output = List.nth outs (n - 1);
    rings = [];
    loop = None;
  }

(* An event as integers: its stage, the row of its window it reads, its pixels,
   and the bases it names, the read's (one, or one per pixel when [reads] is
   positive) then the write's. *)
module Call = struct
  type t = {
    stage : int;
    tap : int;
    pixels : int64;
    reads : int;
    bases : int64 list;
  }

  (* what two calls must share to be the same body *)
  let shape c = (c.stage, c.tap, c.pixels, c.reads)
end

(* Blocks: the calls up to and including each event of the last stage. *)
let blocks calls ~last =
  let rec go cur acc = function
    | [] -> List.rev (if cur = [] then acc else List.rev cur :: acc)
    | (c : Call.t) :: rest ->
        if c.Call.stage = last then go [] (List.rev (c :: cur) :: acc) rest
        else go (c :: cur) acc rest
  in
  go [] [] calls

(* The run of blocks that repeats: [iterations] repetitions of [period] blocks
   from block [first], repetition [k] being repetition 0 with every base moved
   by [k] times one fixed step per base. Prefers the most blocks covered, then
   the shorter period, then the earlier start. *)
let find_loop (bl : Call.t list array) =
  let nb = Array.length bl in
  let rep s p = List.concat (Array.to_list (Array.sub bl s p)) in
  let shapes cs = List.map Call.shape cs in
  let bases cs = List.concat_map (fun (c : Call.t) -> c.Call.bases) cs in
  let best = ref None in
  for p = 1 to nb / 2 do
    for s = 0 to nb - (2 * p) do
      let r0 = rep s p and r1 = rep (s + p) p in
      if shapes r0 = shapes r1 then begin
        let b0 = bases r0 in
        let step = List.map2 Int64.sub (bases r1) b0 in
        let ok k =
          s + ((k + 1) * p) <= nb
          &&
          let r = rep (s + (k * p)) p in
          shapes r = shapes r0
          && bases r
             = List.map2
                 (fun b d -> Int64.add b (Int64.mul (Int64.of_int k) d))
                 b0 step
        in
        let rec count k = if ok k then count (k + 1) else k in
        let n = count 1 in
        let cover = n * p in
        let better =
          match !best with
          | None -> true
          | Some (c, bp, _, _) -> cover > c || (cover = c && p < bp)
        in
        if n >= 2 && better then best := Some (cover, p, s, n)
      end
    done
  done;
  Option.map (fun (_, p, s, n) -> (s, p, n)) !best

let program ?(looped = true) ?mutation stages ~width ~rows ~block =
  match
    List.find_opt (fun (s : Stage.t) -> not (supported s.Stage.kernel)) stages
  with
  | Some s -> Error (Unsupported_window s.Stage.kernel.S.name)
  | None ->
      let n = List.length stages in
      let kernels = List.map (fun (s : Stage.t) -> s.Stage.kernel) stages in
      let schedule = S.run kernels ~input:rows ~block in
      let rows_i = S.Count.to_int rows in
      let shape = shapes stages ~width ~rows:rows_i in
      let windows =
        Array.of_list
          (List.mapi
             (fun i w ->
               if i = n - 1 then None
               else
                 Some
                   (S.Window.make ~rows:w ~chunk:(List.nth kernels i).S.chunk))
             schedule.S.windows)
      in
      let out_rows =
        Array.mapi
          (fun i -> function
            | Some w -> S.Count.to_int (S.Window.rows w)
            | None ->
                let _, _, h = shape.(i) in
                h)
          windows
      in
      let bufs = layout stages ~width ~rows:rows_i ~out_rows in
      let all = ids bufs in
      let input = List.hd all in
      let weights = List.filteri (fun i _ -> i >= 1 && i <= n) all in
      let outs = List.filteri (fun i _ -> i > n) all in
      let stage_index name =
        let rec find i = function
          | [] -> invalid_arg "Band_driver: an event of no stage"
          | (s : Stage.t) :: rest ->
              if String.equal s.Stage.kernel.S.name name then i
              else find (i + 1) rest
        in
        find 0 stages
      in
      let stages_a = Array.of_list stages in
      (* the calls of one event, as integers: one over all its pixels, or for
         a window one per output row and row of the window it reads *)
      let call (e : S.event) =
        let i = stage_index e.S.kernel in
        let st = stages_a.(i) in
        let cin = st.Stage.cin in
        let win, wout, _ = shape.(i) in
        let plo, phi = e.S.produced in
        let clo, chi = e.S.consumed in
        let read_row row =
          if i = 0 then Position.Logical row
          else
            Position.Ring (S.Window.ring_row (Option.get windows.(i - 1)) row)
        in
        let write_row row =
          match windows.(i) with
          | Some w -> Position.Ring (S.Window.ring_row w row)
          | None -> Position.Logical row
        in
        let write row =
          Position.element (write_row row) ~width:wout ~channels:st.Stage.cout
        in
        if S.Count.to_int st.Stage.kernel.S.extent > 1 then
          List.concat_map
            (fun r ->
              List.init (S.Count.to_int st.Stage.kernel.S.extent) (fun tap ->
                  let row =
                    S.Row.of_int
                      (S.Row.to_int (S.Row.of_int r)
                      +
                      match mutation with
                      | Some Window_offset_dropped -> 0
                      | _ -> tap)
                  in
                  {
                    Call.stage = i;
                    tap;
                    pixels = wout;
                    reads = 0;
                    bases =
                      [
                        Position.element (read_row row) ~width:win ~channels:cin;
                        write (S.Row.of_int r);
                      ];
                  }))
            (List.init
               (S.Row.to_int phi - S.Row.to_int plo)
               (fun d -> S.Row.to_int plo + d))
        else
          let pixels =
            Int64.mul (i64 (S.Row.to_int phi - S.Row.to_int plo)) wout
          in
          let per_pixel, reads =
            if i = 0 then
              ( false,
                [
                  Position.element (Position.Logical clo) ~width:win
                    ~channels:cin;
                ] )
            else
              let w = Option.get windows.(i - 1) in
              if S.Window.spans_wrap w ~lo:clo ~hi:chi then
                (* rows that are not consecutive in the window: one base per
                   pixel, each from its own row *)
                ( true,
                  List.concat_map
                    (fun d ->
                      let r = S.Row.of_int (S.Row.to_int clo + d) in
                      List.init (Int64.to_int win) (fun p ->
                          Int64.add
                            (Position.element (read_row r) ~width:win
                               ~channels:cin)
                            (Int64.mul (i64 p) cin)))
                    (List.init (S.Row.to_int chi - S.Row.to_int clo) Fun.id) )
              else
                ( false,
                  [ Position.element (read_row clo) ~width:win ~channels:cin ]
                )
          in
          [
            {
              Call.stage = i;
              tap = 0;
              pixels;
              reads = (if per_pixel then List.length reads else 0);
              bases = reads @ [ write plo ];
            };
          ]
      in
      let calls = List.concat_map call schedule.S.events in
      let bl = Array.of_list (blocks calls ~last:(n - 1)) in
      let found = if looped then find_loop bl else None in
      (* [base] turns one integer offset of the call into an index value *)
      let emit bld (c : Call.t) ~base =
        let i = c.Call.stage in
        let st = stages_a.(i) in
        let src = if i = 0 then input else List.nth outs (i - 1) in
        let rev = List.rev c.Call.bases in
        let write = List.hd rev and read_bases = List.rev (List.tl rev) in
        let read =
          if c.Call.reads > 0 then B.Rows (List.map base read_bases)
          else B.Base (base (List.hd read_bases))
        in
        B.view_call bld
          (desc st ~pixels:c.Call.pixels ~tap:c.Call.tap)
          ~a:(src, read)
          ~b:
            ( List.nth weights i,
              B.Base (B.index bld (tap_weights st ~tap:c.Call.tap)) )
          ~c:(List.nth outs i, B.Base (base write))
      in
      let built =
        Err.or_raise ~pp_error:Ssa_verify.pp_error
          (B.program ~buffers:bufs (fun bld ->
               let run_blocks lo hi =
                 for g = lo to hi - 1 do
                   List.iter
                     (fun c -> emit bld c ~base:(fun x -> B.index bld x))
                     bl.(g)
                 done
               in
               match found with
               | None -> run_blocks 0 (Array.length bl)
               | Some (s, p, iters) ->
                   run_blocks 0 s;
                   let body = List.concat (Array.to_list (Array.sub bl s p)) in
                   let next =
                     List.concat (Array.to_list (Array.sub bl (s + p) p))
                   in
                   let steps =
                     (fun steps ->
                       match mutation with
                       | Some Step_dropped ->
                           List.map (List.map (fun _ -> 0L)) steps
                       | _ -> steps)
                     @@ List.map2
                          (fun (c : Call.t) (c' : Call.t) ->
                            List.map2 Int64.sub c'.Call.bases c.Call.bases)
                          body next
                   in
                   ignore
                     (B.for_ bld ~lo:(B.index bld 0L)
                        ~hi:
                          (B.index bld
                             (i64
                                (match mutation with
                                | Some Iterations_short -> iters - 1
                                | _ -> iters)))
                        ~init:B.Nil
                        (fun b i B.Nil ->
                          List.iter2
                            (fun (c : Call.t) step ->
                              let rest = ref step in
                              emit b c ~base:(fun x ->
                                  let d = List.hd !rest in
                                  rest := List.tl !rest;
                                  if Int64.equal d 0L then B.index b x
                                  else
                                    B.index_add b (B.index b x)
                                      (B.index_scale b d i)))
                            body steps;
                          B.Nil));
                   run_blocks (s + (p * iters)) (Array.length bl)))
      in
      Ok
        {
          Program.program = built;
          input;
          weights;
          output = List.nth outs (n - 1);
          rings =
            List.filter_map
              (fun i ->
                Option.map
                  (fun w -> (List.nth outs i, S.Window.rows w))
                  windows.(i))
              (List.init n Fun.id);
          loop =
            Option.map
              (fun (s, p, iters) ->
                {
                  Loop.first = S.Count.of_int s;
                  period = S.Count.of_int p;
                  iterations = S.Count.of_int iters;
                })
              found;
        }
