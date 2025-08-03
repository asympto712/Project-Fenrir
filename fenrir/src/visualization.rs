use svg::Document;
use svg::Node;
use svg::node::element::{Line, Text};

use std::collections::VecDeque;
use std::rc::Rc;

use rand::prelude::*;

use crate::agent::{MCTSTree, self};
use game::game::GameLogic;
use game::board::TaflBoard;

#[derive(Debug)]
pub struct TreeNode {
    pub visit_count: f32,
    pub children: Vec<Rc<TreeNode>>,
}

impl TreeNode {
    pub fn create_leaf(visit_count: f32) -> Self{
        Self {
            visit_count, // corresponds to the visit count of the edge leading to the node from the upstream
            children: vec![]
        }
    }
    pub fn breadth(&self) -> usize {
        self.children.len()
    }

    pub fn generate<R>(n_child_lim: usize, recurse: usize, rng: &mut R) -> Self
    where R: rand::Rng{
        let n_children = rng.gen_range(0..=n_child_lim);

        // leaf
        if n_children == 0 || recurse == 0 {
            let visit_count: f32 = rng.gen_range(1..=4) as f32;
            Self::create_leaf(visit_count)
        } else {
            let mut children: Vec<Rc<TreeNode>> = vec![];
            for _ in 0..n_children {
                children.push(Rc::new(TreeNode::generate(n_child_lim, recurse - 1, rng)));
            }
            let visit_count = children.iter().fold(0.0, |acc, x| acc + x.visit_count);
            TreeNode{
                visit_count,
                children
            }
        }
    }

    pub fn pre_draw(&self) -> DrawTreeNode {
        if self.children.is_empty() {
            let width = 1;
            DrawTreeNode{
                visit_count: self.visit_count,
                width,
                children: vec![]
            }
        } else {
            let children: Vec<Rc<DrawTreeNode>> = self.children
                .iter()
                .map(|x| {
                    Rc::new(x.pre_draw())
                })
                .collect();
            let width = children
                .iter()
                .fold(0, |acc, x| acc + x.width);

            let width = if width % 2 == 0 { width + 1 }
                else {width};
            
            DrawTreeNode{
                visit_count: self.visit_count,
                width,
                children
            }
        }
    }

    pub fn from_mcts_node<G: GameLogic>(original: &agent::Node<G>) -> Self
    where TaflBoard<<G as GameLogic>::B>: std::fmt::Display {
        let children = original.edges
            .iter()
            .filter(|x| x.as_ref().is_some_and(|edge| agent::Edge::n(&edge) > 0.0))
            .map(|x| {
                let child_rc = agent::Edge::get_child(x.as_ref().unwrap()).clone();
                Rc::new(Self::from_mcts_node(child_rc.as_ref()))
            })
            .collect::<Vec<_>>();

        let visit_count = original.visit_count;

        Self {
            visit_count,
            children
        }
    }
}

#[derive(Debug)]
pub struct Tree {
    pub root: Rc<TreeNode>
}

impl Tree {
    fn depth(&self) -> i64 {
        let mut queue = VecDeque::<(Rc<TreeNode>, i64)>::new();
        let mut max_depth = 0;
        queue.push_back((self.root.clone(), 0));
        while !queue.is_empty() {
            let (popped, depth) = queue.pop_front().unwrap();
            if max_depth < depth { max_depth = depth;}
            for child in popped.children.iter() {
                queue.push_back((child.clone(), depth + 1));
            }
        }
        max_depth
    }

    fn generate(n_child_lim: usize, recurse: usize) -> Self {
        let mut rng = rand::thread_rng();
        let root = TreeNode::generate(n_child_lim, recurse, &mut rng);
        Self { root: Rc::new(root) }
    }

    pub fn pre_draw(&self) -> DrawTree {
        let depth = self.depth();
        let root = self.root.pre_draw();
        DrawTree{
            depth,
            root: Rc::new(root)
        }
    }

    pub fn from_mcts_tree<G: GameLogic>(tree: &MCTSTree<G>) -> Self
    where TaflBoard<<G as GameLogic>::B>: std::fmt::Display {
        let root = TreeNode::from_mcts_node(tree.root.as_ref());
        Self { root: Rc::new(root) }
    }
}

#[derive(Debug)]
pub struct DrawTreeNode {
    visit_count: f32,
    width: i64,
    children: Vec<Rc<DrawTreeNode>>,
}

impl DrawTree {
    pub fn draw(&self, filename: &str, size: XY) {

        let mut document = Document::new().set("viewBox", (0,0,size.x, size.y));
        let dy: f32 = size.y / self.depth as f32;
        let dx: f32 = size.x / self.root.width as f32;

        let mut queue: VecDeque<(Rc<DrawTreeNode>, XY, XY, f32)> = VecDeque::new();
        queue.push_front((
            self.root.clone(),
            XY::new(0.0, 0.0),
            XY::new(dx * self.root.width as f32 / 2.0, 0.0),
            self.root.visit_count,
        ));
        while !queue.is_empty() {
            let (popped, offset, position, visit_count) = queue.pop_front().unwrap();
            let (destinations, _) = popped.children
                .iter()
                .fold((vec![], offset.x),|(mut v, offset_x), node| {
                    let dst = XY::new(
                        offset_x + dx * node.width as f32 / 2.0,
                        offset.y + dy
                    );
                    v.push(dst);
                    (v, offset_x + node.width as f32 * dx)
                });

            let mut offset_x = offset.x;
            for (dst, child) in destinations.into_iter().zip(popped.children.iter()) {

                let stroke_width_ratio: f32 = child.visit_count / visit_count;

                document = document.add(
                    Line::new()
                        .set("x1", position.x)
                        .set("y1", position.y)
                        .set("x2", dst.x)
                        .set("y2", dst.y)
                        .set("stroke", "blue")
                        .set("stroke-width", f32::sqrt(size.x * size.x + size.y * size.y) * 0.01 * stroke_width_ratio)
                );
                queue.push_front((
                    child.clone(),
                    XY::new(offset_x, dst.y),
                    dst,
                    child.visit_count,
                ));

                offset_x += child.width as f32 * dx;
            }
        }
        svg::save(filename, &document).unwrap();
    }
}

#[derive(Debug)]
pub struct DrawTree {
    depth: i64,
    root: Rc<DrawTreeNode>
}

pub struct XY {
    x: f32,
    y: f32,
}

impl XY {
    pub fn new(x: f32, y: f32) -> Self {
        Self{
            x,
            y
        }
    }
}

impl<G: GameLogic> MCTSTree<G>
where TaflBoard<G::B>: std::fmt::Display {
    pub fn draw(&self, filename: &str, size: XY) {
        let tree = Tree::from_mcts_tree(&self);
        let draw_tree = tree.pre_draw();
        draw_tree.draw(filename, size);
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn tree_draw_works() {
        let leaf1 = TreeNode::create_leaf(1.0);
        let leaf2 = TreeNode::create_leaf(2.0);
        let leaf3 = TreeNode::create_leaf(3.0);
        let root = TreeNode {
            visit_count: 6.0,
            children: vec![Rc::new(leaf1), Rc::new(leaf2), Rc::new(leaf3)]
        };
        let tree: Tree = Tree { root: Rc::new(root) };
        assert_eq!(tree.depth(), 1);
        let draw_tree = tree.pre_draw();
        draw_tree.draw("test.svg", XY::new(200.0, 200.0));

    }
}
