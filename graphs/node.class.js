class Node {
    constructor(id,name,edges,_rest) {
        this.id = id;
        this.name = name   || id;
        this.edges = new Map(edges) || new Map(); //Edges is a MAP : Id => Weight
        this.subgraph = null;
        this.type = "Node";
        this.weight=10;
        this._rest = _rest;

    }

    addEdge(to/*Map or edgeId*/,weight/*Boolean or edgeId*/) {
        if (to.type=="Map") {
            // weight is boolean which determines whether tog override original edges
            // true -> Override
            if (weight) this.edges = new Map([...this.edges,...to]);
            else        this.edges = new Map([...to,...this.edges]);
        }
        else  {
            this.edges.set(to,weight);
        }

        return this.edges;
    }
}
