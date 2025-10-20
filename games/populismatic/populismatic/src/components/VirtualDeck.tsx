import {Consumable} from "./Powerup.tsx";
import {Party} from "../flavour.ts";
import Card from "./card/Card.tsx";
import React, {useState} from "react";
import {Faction} from "../Factions.ts";
import {int} from "../modifiers.ts";

type CardAble = Party | Consumable;

function isParty(party: CardAble): party is Party {
    return party?.name != undefined;
}

type DeckProps = {
    expand: (f: Faction) => any,
    onClickPowerup: (c: Consumable, i: number, cb: () => void) => any
    setPowerup: (a: any) => void
    setSelectTiles: (a: any) => void
    initialCards: CardAble[]
}
const sorter = (a: CardAble, b: CardAble) => {
    if (a instanceof Consumable && !(b instanceof Consumable)) return 1;
    if (isParty(a) && isParty(b)) {
        return a.order - b.order;
    }
    if (a instanceof Consumable && b instanceof Consumable) {
        return a.self.cost - b.self.cost;
    }
    return 1;
}
type DeckI = {
    initialCards: CardAble[]
}

export class DeckC {
    deck: CardAble[];
    draw: CardAble[];
    hand: CardAble[];
    handsize: int;
    selected: CardAble | null;

    constructor(initialCards: CardAble[]) {
        this.deck = initialCards;
        this.draw = [];
        this.hand = [];
        this.selected = null;
        this.handsize = 4;
    }

    playCard(i: int) {
        if (this.draw.length + this.deck.length <= this.handsize) {
            // do nothing
        } else {
            this.draw = this.draw.concat(this.hand.filter((_, ii) => ii == i)).slice(1)
            this.hand = this.hand.filter((_, ii) => ii != i).concat(this.draw[0]);
        }
        this.selected = null;
    }

    useCard(i: int) {
        if (this.draw.length + this.hand.length <= this.handsize) {
            // do nothing
        } else {
            this.draw = this.draw.slice(1);
            this.hand = this.hand.filter((_, ii) => ii != i).concat(this.draw[0]);

        }
        this.selected = null;
    }

    set replace(newCards: CardAble[]) {
        this.deck = newCards;
    }
}

export function Deck({initialCards, expand, onClickPowerup, setPowerup, setSelectTiles}: DeckProps) {

    const scards = initialCards;
    const handSize = 4;
    const [cards, setCards] = useState(scards.slice(0, handSize));
    const [drawPile, setDrawPile] = useState(scards.slice(handSize));
    const sortedCards = cards.sort(sorter)
    const [selected, setSelected] = useState<number | null>(null);

    const playCard = (i) => {
        if (drawPile.length + cards.length <= handSize) {
            // do nothing
        } else {
            setDrawPile(drawPile.concat(cards.filter((_, ii) => ii == i)).slice(1));
            setCards(cards.filter((_, ii) => ii != i).concat(drawPile[0]));
        }
        setSelected(null);
    }
    const useCard = (i) => {
        if (drawPile.length + cards.length <= handSize) {
            // do nothing
        } else {
            setDrawPile(drawPile.slice(1));
            setCards(cards.filter((_, ii) => ii != i).concat(drawPile[0]));

        }
        setSelected(null);
    }


    return <>
        <div id={"decktainer"}>
            Hand Size <span>{handSize}</span><br/>
            Deck <span>{cards.length}</span><br/>
            Draw Pile <span>{drawPile.length}</span>
        </div>
        {sortedCards.map((card, i) => {
            if (card instanceof Consumable) {
                return <Card
                    onClick={() => {
                        if (card.self.boardInteraction) {
                            onClickPowerup(card, i, () => useCard(i))
                            setSelected(i);
                        } else {
                            useCard(i);
                        }


                    }}
                    toggle={() => {
                        setPowerup(undefined)
                        setSelectTiles(undefined)
                    }}
                    icon={card.self.icon}
                    title={card.self.name}
                    total={cards.length}
                    nth={i}
                    key={i}
                    bg={card.self.bg}
                    cost={card.self.cost}
                    selected={i === selected}
                    faction={card.self.factions}
                >
                    <>
                        <br/>
                        {card.self.description}</>
                </Card>
            } else if (isParty(card)) {
                return <Card
                    nth={i}
                    key={i}
                    total={cards.length}
                    bg={card.color}
                    title={card.name}
                    cost={1}
                    faction={[card.faction]}
                    selected={i === selected}
                    icon={"👀"} onClick={() => {
                    playCard(i)
                    expand(card.faction)
                }}>
                    <>
                        <b>Steal policy from this party</b>
                        <br/>
                        {card.name}</>
                </Card>
            } else {
                console.log("unknown type", card);
            }
        })}
    </>

}