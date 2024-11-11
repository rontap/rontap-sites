import {Board, Run} from "./Game.ts";
import {Faction, Ideologies} from "./Factions.ts";
import React, {Dispatch, SetStateAction} from "react";
import {Cell, Kind} from "./Cell.ts";
import Effect, {Affect} from "./Effects.tsx";

export type PowerupCtr = {
    name: string,
    icon: string,
    description: string,
    onAction: (cell: Cell | undefined, board: Board, update: Dispatch<SetStateAction<number>>, nextStep: () => void) => void,
    boardInteraction?: boolean,
    that?: Powerup,
    rarity?: number[],

}

type Restr = Faction | Ideologies;

type ConsumableCtr = PowerupCtr & {
    cost: number,
}


type ActCtr = PowerupCtr & {}
type PowerupJSX = {
    onSelect: (consumable: ConsumableCtr) => void,

}

export class Powerup<T extends PowerupCtr> {
    self: T

    constructor(powerupCtr: T) {
        this.self = powerupCtr;
        this.self.that = this;
    }

    button(props: PowerupJSX) {
        return <button className="powerup" onClick={() => props.onSelect(this.self)}>
            {this.self.icon}
            {this.self.name}
        </button>
    }

    extendedButton(props: PowerupJSX) {
        return <div className={"powerupChoice grid space-between"}>
            {this.button(props)}
            <div className={"ml-2"}>{this.self.description}</div>
            <div className="manouverCost">Cost {this.self.cost} Manouvers.</div>
        </div>
    }
}

export class Consumable extends Powerup<ConsumableCtr> {
    constructor(props: ConsumableCtr) {
        super(props);

    }
}


type AdvisorCtr = PowerupCtr & {
    faction: Faction;
    restriction?: Restr[];
    onAcquire: () => void;
    effects: Effect[];
}

export class Advisor<T extends AdvisorCtr> extends Powerup<T> {
    constructor(props: T) {
        super(props);
    }

    extendedButton(props: PowerupJSX) {
        return <div className={"powerupChoice grid space-between"}>
            {this.button(props)}
            <div className={"ml-2"}>{this.self.description}</div>
            <div className={"ml-2"}>{this.self.effects.map(effect => effect.jsx())}</div>
            <div className="manouverCost">Cost {this.self.cost} Manouvers.</div>
        </div>
    }

    description() {
        return <>
            <div className={"ml-2"}>{this.self.description}</div>
            <div className={"ml-2"}>{this.self.effects.map(effect => effect.jsx())}</div>
        </>
    }

    onAcquireEffect(run: Run) {
        this.self.effects.map(effect => effect.doAction(run))
    }

}

type LeaderCtr = AdvisorCtr & {
    faction: Faction;
    actionDescr: string;
    action: string;
}

export enum LeaderNames {
    Merkel, // Angela Merkel
    MP // Magyar Péter
}

export class Leader extends Advisor<LeaderCtr> {
    constructor(props: LeaderCtr) {
        super(props);
    }
}

export const Leaders = {
    [LeaderNames.Merkel]: new Leader({
        onAcquire(): void {
        },
        description: "Angela Merkel",
        faction: Faction.COMM,
        icon: "🫶",
        name: "Angela Merkel",
        action: "Wir Schaffen Das",
        actionDescr: "Sth Sth Sth",
        effects: [],
        onAction(cell: | undefined, board: Board, update: React.Dispatch<React.SetStateAction<number>>, nextStep: () => void): void {
        }
    }),
    [LeaderNames.MP]: new Leader({
        onAcquire(): void {
        },
        description: "",
        faction: Faction.COMM,
        icon: "🇭🇺",
        name: "Magyar Péter",
        action: "Árad a Tisza",
        actionDescr: "Add a Rally consumable.",
        effects: [
            new Effect(Affect.Parties, -0.5, "weight", [Faction.LIB, Faction.GREEN, Faction.SOC, Faction.COMM]),
            new Effect(Affect.Personal, 2, "weight"),
            new Effect(Affect.Kind, 2, "weight", Kind.ACTIVIST),
        ],
        onAction(cell: | undefined, board: Board, update: React.Dispatch<React.SetStateAction<number>>, nextStep: () => void): void {
        }
    })
}

export const KindDescriptions: Record<any, string> = {
    [Kind.ACTIVIST]: "Recruit Activist, place a bomb which explodex in a + shape when convinced.",
    [Kind.RAINBOW]: "Recruit a dedicated follower, who will join regardless of your current ideology. They just like you. Earns 1/2 Score",
    [Kind.DISENFRANCHISED]: "This voter cannot or will not vote. You dont need them to win the level. Earns 0 Score.",
    [Kind.TACTICAL]: "This guy does not like you, but will vote for you regardless because they dislike the opposition more. They will not propagate your ideology however. Earns 2 Score.",
    [Kind.INFLUENCER]: "Recruit a voter as an influencer. Earns 3 score."
} as const
